use anyhow::Result;
use log::info;
use ndarray::Array2;
use ort::{execution_providers::CUDAExecutionProvider, inputs, session::Session, value::Tensor};

/// The loaded ONNX model, kept alive for the duration of the app.
/// We load it once at startup and reuse it for every recording.
pub struct GenderDetector {
    session: Session,
}

/// The output of a detection run — two probabilities that always add up to 1.0.
pub struct GenderResult {
    /// How confident the model is that the speaker sounds female. (0.0–1.0)
    pub female_prob: f32,
    /// How confident the model is that the speaker sounds male. (0.0–1.0)
    pub male_prob: f32,
}

impl GenderResult {
    /// Returns true if the female probability meets or exceeds the threshold.
    /// e.g. threshold 0.5 means "more likely female than not".
    pub fn is_female(&self, threshold: f32) -> bool {
        self.female_prob >= threshold
    }
}

impl GenderDetector {
    /// Load the ONNX model from disk. Call this once when the app starts.
    /// `model_path` is the full path to gender_detection_int8.onnx.
    pub fn new(model_path: &str) -> Result<Self> {
        let model_bytes = std::fs::read(model_path)?;
        let session = Session::builder()?
            .with_execution_providers([CUDAExecutionProvider::default().build()])?
            .commit_from_memory(&model_bytes)?;
        Ok(Self { session })
    }

    /// Run gender detection on raw audio samples.
    ///
    /// `samples` must be 16 kHz mono f32 — exactly what Handy's recorder produces.
    /// Returns probabilities for female and male.
    pub fn detect(&mut self, samples: &[f32]) -> Result<GenderResult> {
        // Step 1: Normalize the audio.
        // Wav2Vec2 was trained on normalized audio, so we must match that at inference
        // time. Normalization means: shift the audio so its average is 0, and scale it
        // so its spread (standard deviation) is 1. This is what Python's
        // Wav2Vec2FeatureExtractor does before running the model.
        let normalized = normalize(samples);

        // Step 2: Reshape into a 2D array with shape [1, num_samples].
        // The model always expects a "batch" dimension — even if we only have one
        // recording. So we wrap it in a batch of size 1.
        let input_array = Array2::from_shape_vec((1, normalized.len()), normalized)?;

        // Step 3: Wrap the array into an ort Tensor value.
        let input_tensor = Tensor::<f32>::from_array(input_array)?;

        // Step 4: Run the model.
        let outputs = self.session.run(inputs![
            "input_values" => input_tensor,
        ])?;

        // Step 6: Pull out the logits from the model's output.
        // "logits" is the standard name for the raw output of a classification model
        // before they've been converted to probabilities. Shape: [1, 2].
        // outputs[0] is female logit, outputs[1] is male logit.
        // In this version of ort, try_extract_tensor returns a (Shape, &[T]) tuple.
        // The flat slice has [female_logit, male_logit] at indices 0 and 1.
        let (_shape, logits) = outputs["logits"].try_extract_tensor::<f32>()?;
        let female_logit = logits[0];
        let male_logit = logits[1];

        info!(
            "Gender logits — female: {:.3}, male: {:.3}",
            female_logit, male_logit
        );

        // Step 6: Apply softmax to convert logits into probabilities.
        // Logits are unbounded raw scores. Softmax squashes them into the range
        // [0.0, 1.0] and makes them sum to exactly 1.0, so they read as percentages.
        let (female_prob, male_prob) = softmax(female_logit, male_logit);

        info!(
            "Gender probs — female: {:.1}%, male: {:.1}%",
            female_prob * 100.0,
            male_prob * 100.0
        );

        Ok(GenderResult {
            female_prob,
            male_prob,
        })
    }
}

/// Normalize audio to zero mean and unit variance.
/// Matches the preprocessing done by Wav2Vec2FeatureExtractor in Python.
fn normalize(samples: &[f32]) -> Vec<f32> {
    let len = samples.len() as f32;
    let mean = samples.iter().sum::<f32>() / len;
    let variance = samples.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / len;
    let std = variance.sqrt();
    // The tiny 1e-7 epsilon prevents dividing by zero if the recording is silence.
    samples.iter().map(|x| (x - mean) / (std + 1e-7)).collect()
}

/// Convert two raw logit scores into probabilities that sum to 1.0.
/// We subtract the max value first — this is a standard numerical stability trick
/// that prevents exp() from producing very large numbers (which would cause NaN).
fn softmax(a: f32, b: f32) -> (f32, f32) {
    let max = a.max(b);
    let exp_a = (a - max).exp();
    let exp_b = (b - max).exp();
    let sum = exp_a + exp_b;
    (exp_a / sum, exp_b / sum)
}
