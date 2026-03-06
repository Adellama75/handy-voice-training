/// Estimates the second formant frequency (F2) from a mono 16 kHz recording.
///
/// Uses LPC (Linear Predictive Coding) to model the vocal tract as a resonant
/// tube, then finds spectral peaks corresponding to formants.
///
/// F2 is the second-lowest formant frequency and is strongly correlated with
/// perceived vocal femininity — female vocal tracts produce higher F2 due to
/// forward tongue placement and shorter vocal tract length. It is the primary
/// trainable feature for voice feminisation.
///
/// Returns the median F2 in Hz across voiced frames, or None if no suitable
/// frames were found (e.g. silence or fully unvoiced speech).
pub fn detect_f2(samples: &[f32], sample_rate: u32) -> Option<f32> {
    // 25 ms frames, 10 ms hop — standard for formant analysis.
    let frame_size = (sample_rate as f32 * 0.025) as usize;
    let hop_size = (sample_rate as f32 * 0.010) as usize;

    // LPC order: rule of thumb is 2 + sample_rate/1000.
    // For 16 kHz that would be 18, but 14 gives cleaner formant peaks
    // by not over-fitting short-term periodicity.
    const LPC_ORDER: usize = 14;

    // Points at which to evaluate the LPC power spectrum (higher = finer resolution).
    const N_SPECTRUM: usize = 1024;

    // Minimum RMS energy to consider a frame voiced (skip silence / stop gaps).
    // Matches the pitch detector's threshold.
    const MIN_ENERGY: f32 = 1e-6;

    let mut f2_values: Vec<f32> = Vec::new();
    let mut start = 0;

    while start + frame_size <= samples.len() {
        let frame = &samples[start..start + frame_size];

        // Skip low-energy frames.
        let energy: f32 = frame.iter().map(|x| x * x).sum::<f32>() / frame_size as f32;
        if energy < MIN_ENERGY {
            start += hop_size;
            continue;
        }

        // Pre-emphasis: apply a first-order high-pass filter to compensate for
        // the -6 dB/octave slope of the glottal source, so all formants have
        // comparable amplitude in the LPC spectrum.
        let mut emphasized = vec![0.0f32; frame_size];
        emphasized[0] = frame[0];
        for i in 1..frame_size {
            emphasized[i] = frame[i] - 0.97 * frame[i - 1];
        }

        // Hamming window: taper the frame edges to reduce spectral leakage.
        for (i, s) in emphasized.iter_mut().enumerate() {
            let w = 0.54
                - 0.46
                    * (2.0 * std::f32::consts::PI * i as f32 / (frame_size - 1) as f32).cos();
            *s *= w;
        }

        // Autocorrelation of the windowed frame — input to Levinson-Durbin.
        let mut r = vec![0.0f32; LPC_ORDER + 1];
        for k in 0..=LPC_ORDER {
            r[k] = (0..frame_size - k)
                .map(|i| emphasized[i] * emphasized[i + k])
                .sum();
        }

        if r[0] < 1e-10 {
            start += hop_size;
            continue;
        }

        if let Some(lpc) = levinson_durbin(&r, LPC_ORDER) {
            let formants = find_formant_peaks(&lpc, sample_rate, N_SPECTRUM);
            // F2 is the second formant (0-indexed: index 1).
            if formants.len() >= 2 {
                f2_values.push(formants[1]);
            }
        }

        start += hop_size;
    }

    if f2_values.is_empty() {
        return None;
    }

    // Median is more robust than mean against frames where the LPC picked up
    // a spurious peak instead of the true F2.
    f2_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Some(f2_values[f2_values.len() / 2])
}

/// Levinson-Durbin recursion: solves the Yule-Walker equations to produce LPC
/// coefficients from autocorrelation values.
///
/// `r` is [r_0, r_1, …, r_order].
/// Returns [a_1, …, a_order] (a_0 = 1 is implicit in the all-pole model).
fn levinson_durbin(r: &[f32], order: usize) -> Option<Vec<f32>> {
    if r[0] < 1e-10 {
        return None;
    }

    // a[i] stores a_{i+1} (0-indexed storage of the 1-indexed LPC coefficients).
    let mut a = vec![0.0f32; order];
    let mut a_prev = vec![0.0f32; order];
    let mut alpha = r[0]; // Running prediction-error energy.

    for k in 1..=order {
        // Reflection coefficient for step k:
        // lambda = r_k + sum_{j=1}^{k-1} a_j * r_{k-j}
        // In 0-indexed: r[k] + sum_{j=0}^{k-2} a[j] * r[k-1-j]
        let mut lambda = r[k];
        for j in 0..(k - 1) {
            lambda += a[j] * r[k - 1 - j];
        }

        let km = (-lambda / alpha).clamp(-1.0, 1.0);

        // Update coefficients: a_j_new = a_j + km * a_{k-j} for j = 1..k-1.
        // In 0-indexed: a[j] = a_prev[j] + km * a_prev[k-2-j] for j = 0..k-2.
        a_prev[..k - 1].copy_from_slice(&a[..k - 1]);
        for j in 0..(k - 1) {
            a[j] = a_prev[j] + km * a_prev[k - 2 - j];
        }
        a[k - 1] = km;

        alpha *= 1.0 - km * km;
        if alpha <= 1e-10 {
            break;
        }
    }

    Some(a)
}

/// Evaluate the LPC power spectrum and return formant frequencies as local maxima.
///
/// The synthesis filter is H(z) = 1 / A(z) where
///   A(e^{jω}) = 1 + a_1 e^{-jω} + a_2 e^{-j2ω} + … + a_p e^{-jpω}
///
/// Formants are the poles of H, visible as peaks in |H(e^{jω})|² = 1/|A(e^{jω})|².
/// We evaluate at `n_points` evenly-spaced frequencies from 0 to Nyquist.
fn find_formant_peaks(lpc: &[f32], sample_rate: u32, n_points: usize) -> Vec<f32> {
    let nyquist = sample_rate as f32 / 2.0;
    let mut spectrum = vec![0.0f32; n_points];

    for i in 0..n_points {
        let freq = i as f32 * nyquist / n_points as f32;
        let omega = 2.0 * std::f32::consts::PI * freq / sample_rate as f32;
        // A(e^{jω}) = 1 + sum_{k=1}^{p} a_k * e^{-jkω}
        let mut re = 1.0f32;
        let mut im = 0.0f32;
        for (k, &a_k) in lpc.iter().enumerate() {
            let angle = -((k + 1) as f32) * omega;
            re += a_k * angle.cos();
            im += a_k * angle.sin();
        }
        let mag_sq = re * re + im * im;
        spectrum[i] = if mag_sq > 1e-10 { 1.0 / mag_sq } else { 0.0 };
    }

    // Search for peaks only between 90 Hz (below F0 territory) and 5 kHz.
    let min_bin = (90.0 * n_points as f32 / nyquist) as usize;
    let max_bin = ((5000.0f32).min(nyquist) * n_points as f32 / nyquist) as usize;
    let max_bin = max_bin.min(n_points - 2);

    // Significance threshold: reject tiny bumps that are < 5 % of the dominant peak.
    let max_val = spectrum[min_bin..=max_bin]
        .iter()
        .cloned()
        .fold(0.0f32, f32::max);
    let threshold = max_val * 0.02;

    let mut peaks: Vec<f32> = Vec::new();
    for i in min_bin.max(1)..=max_bin {
        if spectrum[i] > spectrum[i - 1]
            && spectrum[i] > spectrum[i + 1]
            && spectrum[i] > threshold
        {
            peaks.push(i as f32 * nyquist / n_points as f32);
        }
    }

    // Peaks are already in ascending frequency order.
    peaks
}
