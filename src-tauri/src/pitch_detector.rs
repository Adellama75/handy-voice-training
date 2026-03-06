/// Estimates the fundamental frequency (F0 / pitch) of a mono 16kHz audio recording.
///
/// Uses autocorrelation: we slide a copy of the signal over itself and find the
/// delay (lag) at which it matches itself best. That lag corresponds to one period
/// of the repeating waveform, so F0 = sample_rate / lag.
///
/// Returns the median F0 across all voiced frames in Hz, or None if no voiced
/// frames were found (e.g. silence or unvoiced speech).
pub fn detect_pitch(samples: &[f32], sample_rate: u32) -> Option<f32> {
    // Frame size: 25ms — long enough to capture several cycles of the lowest
    // expected pitch (50 Hz = 20ms period), short enough to track changes.
    let frame_size = (sample_rate as f32 * 0.025) as usize;

    // Hop size: 10ms between frames.
    let hop_size = (sample_rate as f32 * 0.010) as usize;

    // Search range: 50 Hz–500 Hz covers all human voices.
    // lag = sample_rate / frequency, so:
    //   max_lag = sample_rate / min_freq  (low freq = long period = large lag)
    //   min_lag = sample_rate / max_freq  (high freq = short period = small lag)
    let min_lag = (sample_rate as f32 / 500.0) as usize;
    let max_lag = (sample_rate as f32 / 50.0) as usize;

    let mut frame_pitches: Vec<f32> = Vec::new();

    let mut start = 0;
    while start + frame_size <= samples.len() {
        let frame = &samples[start..start + frame_size];

        if let Some(f0) = autocorrelate(frame, min_lag, max_lag, sample_rate) {
            frame_pitches.push(f0);
        }

        start += hop_size;
    }

    if frame_pitches.is_empty() {
        return None;
    }

    // Use the median rather than the mean — it's much more robust to the
    // occasional frame where the autocorrelation picked up a harmonic instead
    // of the true fundamental.
    frame_pitches.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(frame_pitches[frame_pitches.len() / 2])
}

/// Run autocorrelation on a single frame and return the F0 if voiced.
///
/// Autocorrelation at lag L = sum of (x[i] * x[i+L]) for all i.
/// A high value means the signal repeats with period L.
/// We normalize so the result is 1.0 at lag 0 (perfect match with itself).
fn autocorrelate(frame: &[f32], min_lag: usize, max_lag: usize, sample_rate: u32) -> Option<f32> {
    let n = frame.len();

    // r(0) = energy of the frame — used for normalization.
    let r0: f32 = frame.iter().map(|x| x * x).sum();

    // Silence check: if the frame has almost no energy, skip it.
    // This avoids picking up "pitch" from noise between words.
    if r0 < 1e-6 {
        return None;
    }

    // Compute autocorrelation for each candidate lag.
    let max_lag = max_lag.min(n - 1);
    let mut best_lag = min_lag;
    let mut best_r = f32::NEG_INFINITY;

    for lag in min_lag..=max_lag {
        let r: f32 = (0..n - lag).map(|i| frame[i] * frame[i + lag]).sum();
        let normalized = r / r0;
        if normalized > best_r {
            best_r = normalized;
            best_lag = lag;
        }
    }

    // Voicing threshold: if the best correlation is below 0.3, the frame is
    // likely unvoiced (fricatives like "s", "f") or silence. Skip it.
    if best_r < 0.3 {
        return None;
    }

    Some(sample_rate as f32 / best_lag as f32)
}
