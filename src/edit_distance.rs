pub fn compare(string: &str, other: &str, max_distance: i64) -> Option<i64> {
    use triple_accel::levenshtein::{levenshtein_simd_k_with_opts, RDAMERAU_COSTS};
    let dist = levenshtein_simd_k_with_opts(
        string.as_ref(),
        other.as_ref(),
        max_distance as u32,
        false,
        RDAMERAU_COSTS,
    );

    dist.map(|v| v.0 as i64)
}