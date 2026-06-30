use itertools::Itertools;
use std::{io::{Error, ErrorKind}, str::FromStr, default::Default};

/// Background nucleotide probabilities in A/C/G/T order.
///
/// These probabilities define the null model used by motif scanners. Match
/// scores are computed as natural-log likelihood ratios against this
/// background, i.e. each aligned base contributes `ln(P_motif(base) /
/// P_background(base))`.
#[derive(Debug, Clone)]
pub struct BackgroundProb(pub [f64; 4]);

impl Default for BackgroundProb {
    fn default() -> Self {
        BackgroundProb([0.25, 0.25, 0.25, 0.25])
    }
}

/// A DNA motif represented by position-specific A/C/G/T probabilities.
///
/// The rows of `probability` are expected to be ordered as A, C, G, T. When a
/// motif is converted to a [`DNAMotifScanner`], pseudocounts are added to zero
/// probabilities and rows are normalized before constructing the score
/// distribution.
#[derive(Debug, Clone)]
pub struct DNAMotif {
    pub id: String,
    pub name: Option<String>,
    pub family: Option<String>,
    pub probability: Vec<[f64; 4]>,
}

impl FromStr for DNAMotif {
    type Err = Error;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut lines = s.lines();
        let first_line = lines.next().unwrap();
        if &first_line[..5] != "MOTIF" {
            return Err(Error::new(ErrorKind::Other, "MOTIF not found"));
        }
        // skip
        lines.next();
        let pwm = lines.map(|x| x.trim().split_ascii_whitespace()
            .map(|v| v.parse().unwrap()).collect::<Vec<_>>().try_into().unwrap()
        ).collect();

        Ok(DNAMotif {
            id: first_line.strip_prefix("MOTIF ").unwrap().to_string(),
            name: None,
            family: None,
            probability: pwm,
        })
    }
}

impl DNAMotif {
    pub fn size(&self) -> usize { self.probability.len() }

    pub fn info_content(&self) -> f64 {
        self.probability.iter().map(|row| {
            let entropy: f64 = row.into_iter().map(|p| if *p == 0.0 {
                0.0
            } else {
                -1.0 * *p * p.log2()
            }).sum();
            2.0 - entropy
        }).sum()
    }

    /// Create a scanner using the supplied background probabilities.
    ///
    /// The scanner reports motif hit scores on the natural-log likelihood-ratio
    /// scale, not p-values. The p-value supplied to [`DNAMotifScanner::find`] is
    /// used only to derive the minimum score threshold from the scanner's score
    /// distribution under the background model.
    pub fn to_scanner(mut self, bg: BackgroundProb) -> DNAMotifScanner {
        self.add_pseudocount(0.0001);
        let cdf = ScoreCDF::new(&self, &bg);
        DNAMotifScanner {
            motif: self,
            cdf,
            background: bg,
        }
    }

    pub fn revcomp(&self) -> Self {
        todo!()
    }

    fn add_pseudocount(&mut self, pseudocount: f64) {
        self.probability.iter_mut().for_each(|ps| {
            ps.iter_mut().for_each(|p| if *p == 0.0 { *p = pseudocount; });
            let s: f64 = ps.iter().sum();
            if s != 1.0 {
                ps.iter_mut().for_each(|p| *p /= s);
            }
        });
    }

    fn optimal_scores_suffix(&self, bg: &BackgroundProb) -> Vec<f64> {
        let mut scores: Vec<f64> = self.probability.iter().scan(0.0, |state, prob| {
            let (i, p) = prob.iter().enumerate()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
            *state = *state + (p / bg.0[i]).ln();
            Some(*state)
        }).collect();
        let max = *scores.last().unwrap();
        scores.iter_mut().for_each(|x| *x = max - *x);
        scores
    }

    // This function does not do bound checks on seq.
    // Returned scores are natural-log likelihood ratios, not p-values.
    fn look_ahead_search(
        &self,
        bg: &BackgroundProb,
        remain_best: &Vec<f64>,  // best possible match score of suffixes
        seq: &[u8],
        start: usize,
        thres: f64,
    ) -> Option<(usize, f64)> {
        let n = self.size();
        let mut cur_pos = 0;
        let mut cur_match = 0.0;
        loop {
            let sc = match seq[cur_pos + start] {
                b'A' | b'a' => (self.probability[cur_pos][0] / bg.0[0]).ln(),
                b'C' | b'c' => (self.probability[cur_pos][1] / bg.0[1]).ln(),
                b'G' | b'g' => (self.probability[cur_pos][2] / bg.0[2]).ln(),
                b'T' | b't' => (self.probability[cur_pos][3] / bg.0[3]).ln(),
                b'N' | b'n' => 0.0,
                _ => panic!("invalid nucleotide: {}", String::from_utf8(vec![seq[cur_pos + start]]).unwrap()),
            };
            cur_match += sc;
            let cur_best = cur_match + remain_best[cur_pos];

            if cur_best < thres {
                return None;
            } else if cur_pos >= n - 1 {
                return Some((start, cur_best));
            } else {
                cur_pos += 1;
            }
        }
    }
}

/// Scanner for finding motif hits in DNA sequences.
///
/// Hits are filtered by a p-value threshold but reported as `(position, score)`.
/// The reported `score` is the natural-log likelihood ratio of the aligned
/// sequence window under the motif model versus the background model:
///
/// `sum_i ln(P_motif_i(base_i) / P_background(base_i))`
///
/// It is not a p-value and it is not a log-transformed p-value. The p-value
/// argument to [`find`](DNAMotifScanner::find) controls the minimum score by
/// converting the upper-tail probability into a score threshold through the
/// precomputed score CDF.
#[derive(Debug, Clone)]
pub struct DNAMotifScanner {
    pub motif: DNAMotif,
    cdf: ScoreCDF,
    background: BackgroundProb,
}

/// A motif scan hit.
///
/// `position` is the zero-based start position of the hit in the scanned
/// sequence. `score` is the natural-log likelihood-ratio motif score. When
/// requested by [`DNAMotifScanner::find`], `log10_p_value` contains the
/// upper-tail p-value estimated from the scanner's score CDF on a log10 scale.
/// Otherwise it is `None`.
#[derive(Debug, Clone)]
pub struct MotifScanResult {
    pub position: usize,
    pub score: f64,
    pub log10_p_value: Option<f64>,
}

impl DNAMotifScanner {
    /// Find forward-strand motif hits in `seq` at or above the p-value cutoff.
    ///
    /// The `pvalue` argument is interpreted as an upper-tail probability under
    /// the scanner's background model. Internally it is converted to a score
    /// cutoff with `cdf.inverse(1 - pvalue)`, and only windows with scores at or
    /// above that cutoff are returned.
    ///
    /// Returned items contain the zero-based position in `seq`, the natural-log
    /// likelihood-ratio motif match score, and an optional log10 p-value. The
    /// score is not a p-value and is not on a log p-value scale.
    ///
    /// # Notes
    ///
    /// For long or information-rich motifs, returned hits will usually have
    /// reported p-values below the user-defined cutoff. For short or degenerate
    /// motifs, the best possible motif score may still have an estimated p-value
    /// larger than the requested cutoff. In this case, `find` may return exact
    /// or best-possible matches whose reported p-value is larger than `pvalue`.
    /// This behavior is intentional: the `pvalue` argument is used to derive a
    /// score cutoff, and the scanner keeps best-possible matches reachable even
    /// when they are not statistically significant under the motif score CDF. If
    /// strict p-value filtering is required, call `find` with `report_pvalue` set
    /// to `true` and filter returned hits by the reported log10 p-value.
    pub fn find<'a>(
        &'a self,
        seq: &'a [u8],
        pvalue: f64,
        report_pvalue: bool,
    ) -> MotifSites<'a>
    {
        let thres = self.cdf.prob_inverse(1.0 - pvalue);
        MotifSites {
            motif: &self.motif,
            cdf: &self.cdf,
            sigma: self.motif.optimal_scores_suffix(&self.background),
            background: &self.background,
            seq: seq,
            cur_pos: 0,
            thres,
            report_pvalue,
        }
    }
}

/// Approximate CDF of motif match scores under the background model.
///
/// Scores in this CDF use the same natural-log likelihood-ratio scale returned
/// by [`DNAMotifScanner::find`]. The CDF maps score thresholds to cumulative
/// background probability `P(score <= threshold)`.
#[derive(Debug, Clone)]
struct ScoreCDF(Vec<(f64, f64)>);

impl ScoreCDF {
    /// Approximate the CDF of motif matching scores using dynamic programming.
    ///
    /// The score for a sequence window is a natural-log likelihood ratio against
    /// the background model. For each motif position, the possible base scores
    /// are `ln(P_motif(base) / P_background(base))`; dynamic programming then
    /// accumulates the background probability mass for binned total scores.
    fn new(motif: &DNAMotif, bg: &BackgroundProb) -> Self {
        struct ScoreGetter {
            lowest: f64,
            step: f64,
        }
        impl ScoreGetter {
            fn get_sc(&self, i: usize) -> f64 { (i as f64 + 0.5) * self.step + self.lowest }
        }

        let precision = 1e-5;
        let init = (vec![1.0], ScoreGetter { lowest: 0.0, step: 0.0 });
        let (accum, getter) = motif.probability.iter().fold(init, |(accum, getter), probs| {
            let normalized_probs: Vec<f64> = probs.iter().zip(bg.0.iter())
                .map(|(p_fg, p_bg)| (p_fg / p_bg).ln()).collect();
            let (min_prob, max_prob) = normalized_probs.iter()
                .minmax().into_option().unwrap();
            let lowest = getter.get_sc(
                accum.iter().enumerate().skip_while(|(_, x)| **x == 0.0).next().unwrap().0
            ) + min_prob;
            let highest = getter.get_sc(
                accum.iter().enumerate().rev().skip_while(|(_, x)| **x == 0.0).next().unwrap().0
            ) + max_prob;
            if lowest < highest {
                let num_bins = ((highest - lowest) / precision).ceil().min(200000.0) as usize;
                let step = (highest - lowest) / num_bins as f64;
                let mut new_accum = vec![0.0; num_bins];
                accum.into_iter().enumerate().for_each(|(i, v)| if v != 0.0 {
                    let sc = getter.get_sc(i);
                    normalized_probs.iter().zip(bg.0.iter()).for_each(|(p_norm, p_bg)| {
                        let idx = (((sc + p_norm - lowest) / step).floor() as usize)
                            .min(num_bins - 1);
                        new_accum[idx] += v * p_bg;
                    });
                });
                (new_accum, ScoreGetter { lowest, step })
            } else {
                (accum, getter)
            }
        });

        // TODO: compress CDF
        let cdf = accum.into_iter().scan(0.0, |state, x| {
            *state += x;
            Some(*state)
        }).enumerate().map(|(i, x)| (getter.get_sc(i), x))
            .chunk_by(|x| x.1).into_iter().flat_map(|(_, mut groups)| {
                let a = groups.next().unwrap();
                match groups.last() {
                    None => vec![a],
                    Some(b) => vec![a, b],
                }
            }).collect();
        ScoreCDF(cdf)
    }

    /// Return the score threshold corresponding to cumulative probability `p`.
    ///
    /// Callers that want an upper-tail p-value cutoff should pass `1 - pvalue`.
    /// The returned value is a natural-log likelihood-ratio score threshold.
    fn prob_inverse(&self, p: f64) -> f64 {
        if p > 1.0 || p < 0.0 {
            panic!("p must be in [0,1]");
        }
        let cdf = &self.0;
        let n = cdf.len();
        let i = cdf.binary_search_by(|x| x.1.partial_cmp(&p).unwrap())
            .unwrap_or_else(|x| x);
        if i >= n {
            panic!("impossible");
        } else if i == 0 {
            if p == cdf[0].1 {
                cdf[0].0
            } else {
                panic!("impossible");
            }
        } else {
            let (ix_a, p_a) = cdf[i-1];
            let (ix_b, p_b) = cdf[i];
            let w1 = (p_b - p) / (p_b - p_a);
            let w2 = (p - p_a) / (p_b - p_a);
            w1 * ix_a + w2 * ix_b
        }
    }

    /// Return `log10(P(score >= threshold))` estimated from the score CDF.
    fn log10_upper_tail_prob(&self, threshold: f64) -> f64 {
        let cdf = &self.0;
        let n = cdf.len();
        if n == 0 {
            panic!("empty score CDF");
        }
        let upper_tail = match cdf.binary_search_by(|x| x.0.partial_cmp(&threshold).unwrap()) {
            Ok(i) => 1.0 - if i == 0 { 0.0 } else { cdf[i-1].1 },
            Err(0) => 1.0,
            Err(i) if i >= n => {
                if n == 1 { 1.0 } else { 1.0 - cdf[n-2].1 }
            },
            Err(i) => {
                let (score_a, p_a) = cdf[i-1];
                let (score_b, p_b) = cdf[i];
                let w1 = (score_b - threshold) / (score_b - score_a);
                let w2 = (threshold - score_a) / (score_b - score_a);
                1.0 - (w1 * p_a + w2 * p_b)
            },
        };
        let upper_tail = upper_tail.clamp(0.0, 1.0);
        if upper_tail == 0.0 {
            f64::NEG_INFINITY
        } else {
            upper_tail.log10()
        }
    }
}

pub struct MotifSites<'a> {
    motif: &'a DNAMotif,
    cdf: &'a ScoreCDF,
    sigma: Vec<f64>,
    background: &'a BackgroundProb,
    seq: &'a [u8],
    cur_pos: usize,
    thres: f64,
    report_pvalue: bool,
}

impl<'a> Iterator for MotifSites<'a> {
    type Item = MotifScanResult;

    /// Return the next motif hit.
    ///
    /// `score` is the natural-log likelihood-ratio match score for the motif
    /// window. It is not the p-value used to filter hits. `log10_p_value` is
    /// populated only when requested by `DNAMotifScanner::find`.
    fn next(&mut self) -> Option<Self::Item> {
        loop {
            if self.cur_pos + self.motif.size() >= self.seq.len() + 1 {
                return None;
            }
            let search_result = self.motif.look_ahead_search(
                self.background,
                &self.sigma,
                self.seq,
                self.cur_pos,
                self.thres,
            );
            self.cur_pos += 1;
            if let Some((position, score)) = search_result {
                let log10_p_value = if self.report_pvalue {
                    Some(self.cdf.log10_upper_tail_prob(score))
                } else {
                    None
                };
                return Some(MotifScanResult { position, score, log10_p_value });
            }
        }
    }
}

pub fn parse_meme(content: &str) -> Vec<DNAMotif> {
    content.split("MOTIF").skip(1).map(|s| {
        let mut lines = s.lines();
        let id = lines.next().unwrap().trim().to_string();
        let mut iter = lines.skip_while(|x| !x.starts_with("letter-probability matrix"));
        let n: usize = iter.next().unwrap().split("w=").last().unwrap()
            .split("nsites=").next().unwrap().trim().parse().unwrap();
        let pwm = iter.take(n).map(|x| x.trim().split_ascii_whitespace()
            .map(|v| v.parse().unwrap()).collect::<Vec<_>>().try_into().unwrap()
        ).collect();
        DNAMotif { id, name: None, family: None, probability: pwm }
    }).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let bg = BackgroundProb::default();
        let motif1_str = "MOTIF 1_ASCCAGGCKGG
letter-probability matrix: alength= 4 w= 11 nsites= 14 E= 3.2e-035
0.768791 0.07577 0.120456 0.034983
0.057276 0.532522375 0.282802875 0.12739862500000002
0.048894125 0.796682375 0.09644475 0.05797825
0.044861750000000006 0.6375725 0.263928 0.05363775
0.61916725 0.104690625 0.17939824999999998 0.096743875
0.086404375 0.0745045 0.807213125 0.031878125
0.08837187499999999 0.07052475 0.828965875 0.012137499999999999
0.039856875 0.80558425 0.08105799999999999 0.07350025
0.079739875 0.110598625 0.40112975 0.40853125
0.03881925 0.12873025 0.781449 0.05100175
0.136913 0.15827100000000002 0.5818685 0.1229465";
        let scores = vec![7.009906220318511];

        let motif1: DNAMotif = motif1_str.parse().unwrap();
        let cdf = ScoreCDF::new(&motif1, &Default::default());
        assert!((cdf.prob_inverse(1.0 - 1e-4) - scores[0]).abs() < 1e-3);
        for pvalue in [1e-2_f64, 1e-4_f64] {
            let score = cdf.prob_inverse(1.0 - pvalue);
            let log10_pvalue = cdf.log10_upper_tail_prob(score);
            assert!((log10_pvalue - pvalue.log10()).abs() < 1e-10);
        }

        let seq = b"ACCCAGGCTGG";
        let hits = motif1
            .clone()
            .to_scanner(bg.clone())
            .find(seq, 0.9, true)
            .collect::<Vec<_>>();
        assert_eq!(hits.len(), 1);
        assert_eq!(hits[0].position, 0);
        assert!(hits[0].log10_p_value.is_some());

        let hits = motif1
            .to_scanner(bg)
            .find(seq, 0.9, false)
            .collect::<Vec<_>>();
        assert_eq!(hits.len(), 1);
        assert!(hits[0].log10_p_value.is_none());
    }

    #[test]
    fn log10_upper_tail_prob_uses_cdf() {
        let cdf = ScoreCDF(vec![(0.0, 0.25), (1.0, 0.75), (2.0, 1.0)]);

        assert_eq!(cdf.log10_upper_tail_prob(-1.0), 0.0);
        assert_eq!(cdf.log10_upper_tail_prob(0.0), 0.0);
        assert!((cdf.log10_upper_tail_prob(1.0) - 0.75_f64.log10()).abs() < 1e-12);
        assert!((cdf.log10_upper_tail_prob(0.5) - 0.5_f64.log10()).abs() < 1e-12);
        assert!((cdf.log10_upper_tail_prob(2.0) - 0.25_f64.log10()).abs() < 1e-12);
        assert!((cdf.log10_upper_tail_prob(3.0) - 0.25_f64.log10()).abs() < 1e-12);
    }
}
