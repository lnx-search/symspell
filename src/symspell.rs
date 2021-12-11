use std::cmp;
use std::fs::File;
use std::i64;
use std::io::{BufRead, BufReader};
use std::ops::Deref;

use hashbrown::{HashMap, HashSet};

use crate::composition::Composition;
use crate::suggestion::Suggestion;
use crate::string_strategy::StringStrategy;
use crate::wordmap::{Word, WordMap};

#[derive(Eq, PartialEq, Debug)]
pub enum Verbosity {
    Top,
    Closest,
    All,
}

pub struct SymSpell<T: StringStrategy> {
    /// Maximum edit distance for doing lookups.
    max_dictionary_edit_distance: i64,
    /// The length of word prefixes used for spell checking.
    prefix_length: i64,
    /// The minimum frequency count for dictionary words to be considered correct spellings.
    count_threshold: i64,

    //// number of all words in the corpus used to generate the
    //// frequency dictionary. This is used to calculate the word
    //// occurrence probability p from word counts c : p=c/N. N equals
    //// the sum of all counts c in the dictionary only if the
    //// dictionary is complete, but not if the dictionary is
    //// truncated or filtered
    corpus_word_count: i64,

    max_length: i64,
    deletes: WordMap,
    words: HashMap<String, i64>,
    string_strategy: T,
}

impl<T: StringStrategy> Default for SymSpell<T> {
    fn default() -> Self {
        Self {
            max_dictionary_edit_distance: 2,
            prefix_length: 7,
            count_threshold: 1,
            corpus_word_count: i64::MAX,
            max_length: 0,
            deletes: Default::default(),
            words: Default::default(),
            string_strategy: T::new()
        }
    }
}

impl<T: StringStrategy> SymSpell<T> {
    /// Find suggested spellings for a given input word, using the maximum
    /// edit distance specified during construction of the SymSpell dictionary.
    ///
    /// # Arguments
    ///
    /// * `input` - The word being spell checked.
    /// * `verbosity` - The value controlling the quantity/closeness of the retuned suggestions.
    /// * `max_edit_distance` - The maximum edit distance between input and suggested words.
    pub fn lookup(
        &self,
        input: &str,
        verbosity: Verbosity,
        max_edit_distance: i64,
    ) -> Vec<Suggestion> {
        if max_edit_distance > self.max_dictionary_edit_distance {
            panic!("max_edit_distance is bigger than max_dictionary_edit_distance");
        }

        let mut suggestions: Vec<Suggestion> = Vec::new();

        let prep_input = self.string_strategy.prepare(input);
        let input = prep_input.as_str();
        let input_len = self.string_strategy.len(input) as i64;

        if input_len - self.max_dictionary_edit_distance > self.max_length {
            return suggestions;
        }

        let mut hashset1: HashSet<String> = HashSet::new();
        let mut hashset2: HashSet<Word> = HashSet::new();

        if self.words.contains_key(input) {
            let suggestion_count = self.words[input];
            suggestions.push(Suggestion::new(input, 0, suggestion_count));

            if verbosity != Verbosity::All {
                return suggestions;
            }
        }

        hashset2.insert(Word::from(input));

        let mut max_edit_distance2 = max_edit_distance;
        let mut candidate_pointer = 0;
        let mut candidates = Vec::new();

        let mut input_prefix_len = input_len;

        if input_prefix_len > self.prefix_length {
            input_prefix_len = self.prefix_length;
            candidates.push(
                self.string_strategy
                    .slice(input, 0, input_prefix_len as usize),
            );
        } else {
            candidates.push(input.to_string());
        }

        while candidate_pointer < candidates.len() {
            let candidate = &candidates.get(candidate_pointer).unwrap().clone();
            candidate_pointer += 1;
            let candidate_len = self.string_strategy.len(candidate) as i64;
            let length_diff = input_prefix_len - candidate_len;

            if length_diff > max_edit_distance2 {
                if verbosity == Verbosity::All {
                    continue;
                }
                break;
            }

            if let Some(dict_suggestions) = self.deletes.get(&candidate) {
                for suggestion in dict_suggestions {
                    let suggestion_len = self.string_strategy.len(suggestion.as_str()) as i64;

                    if suggestion == input {
                        continue;
                    }

                    if (suggestion_len - input_len).abs() > max_edit_distance2
                        || suggestion_len < candidate_len
                        || (suggestion_len == candidate_len && suggestion.as_str() != candidate)
                    {
                        continue;
                    }

                    let sugg_prefix_len = cmp::min(suggestion_len, self.prefix_length);

                    if sugg_prefix_len > input_prefix_len
                        && sugg_prefix_len - candidate_len > max_edit_distance2
                    {
                        continue;
                    }

                    let distance;

                    if candidate_len == 0 {
                        distance = cmp::max(input_len, suggestion_len);

                        if distance > max_edit_distance2 || hashset2.contains(suggestion) {
                            continue;
                        }
                        hashset2.insert(suggestion.deref().clone());
                    } else if suggestion_len == 1 {
                        distance = if !input.contains(&self.string_strategy.slice(suggestion.as_str(), 0, 1))
                        {
                            input_len
                        } else {
                            input_len - 1
                        };

                        if distance > max_edit_distance2 || hashset2.contains(suggestion) {
                            continue;
                        }

                        hashset2.insert(suggestion.deref().clone());
                    } else if self.has_different_suffix(
                        max_edit_distance,
                        input,
                        input_len,
                        candidate_len,
                        suggestion.as_str(),
                        suggestion_len,
                    ) {
                        continue;
                    } else {
                        if verbosity != Verbosity::All
                            && !self.delete_in_suggestion_prefix(
                                candidate,
                                candidate_len,
                                suggestion.as_str(),
                                suggestion_len,
                            )
                        {
                            continue;
                        }

                        if hashset2.contains(suggestion) {
                            continue;
                        }
                        hashset2.insert(suggestion.deref().clone());

                        if let Some(d) = quick_compare(input, suggestion.as_str(), max_edit_distance2 as u8) {
                            distance = d as i64;
                        } else {
                            continue
                        };
                    }

                    if distance <= max_edit_distance2 {
                        let suggestion_count = self.words[suggestion.as_str()];
                        let si = Suggestion::new(suggestion.to_string(), distance, suggestion_count);

                        if !suggestions.is_empty() {
                            match verbosity {
                                Verbosity::Closest => {
                                    if distance < max_edit_distance2 {
                                        suggestions.clear();
                                    }
                                }
                                Verbosity::Top => {
                                    if distance < max_edit_distance2
                                        || suggestion_count > suggestions[0].count
                                    {
                                        max_edit_distance2 = distance;
                                        suggestions[0] = si;
                                    }
                                    continue;
                                }
                                _ => (),
                            }
                        }

                        if verbosity != Verbosity::All {
                            max_edit_distance2 = distance;
                        }

                        suggestions.push(si);
                    }
                }
            }

            if length_diff < max_edit_distance && candidate_len <= self.prefix_length {
                if verbosity != Verbosity::All && length_diff >= max_edit_distance2 {
                    continue;
                }

                for i in 0..candidate_len {
                    let delete = self.string_strategy.remove(candidate, i as usize);

                    if !hashset1.contains(&delete) {
                        hashset1.insert(delete.clone());
                        candidates.push(delete);
                    }
                }
            }
        }

        if suggestions.len() > 1 {
            suggestions.sort();
        }

        suggestions
    }

    /// Find suggested spellings for a given input sentence, using the maximum
    /// edit distance specified during construction of the SymSpell dictionary.
    ///
    /// # Arguments
    ///
    /// * `input` - The sentence being spell checked.
    /// * `max_edit_distance` - The maximum edit distance between input and suggested words.
    pub fn lookup_compound(&self, input: &str, edit_distance_max: i64) -> String {
        //parse input string into single terms
        let term_list1 = self.parse_words(&self.string_strategy.prepare(input));

        // let mut suggestions_previous_term: Vec<Suggestion> = Vec::new();                  //suggestions for a single term
        let mut suggestions: Vec<Suggestion>;
        let mut suggestion_parts: Vec<Suggestion> = Vec::new();

        //translate every term to its best suggestion, otherwise it remains unchanged
        let mut last_combi = false;

        for (i, term) in term_list1.iter().enumerate() {
            suggestions = self.lookup(term, Verbosity::Top, edit_distance_max);

            //combi check, always before split
            if i > 0 && !last_combi {
                let mut suggestions_combi: Vec<Suggestion> = self.lookup(
                    &format!("{}{}", term_list1[i - 1], term_list1[i]),
                    Verbosity::Top,
                    edit_distance_max,
                );

                if !suggestions_combi.is_empty() {
                    let best1 = suggestion_parts[suggestion_parts.len() - 1].clone();
                    let best2 = if !suggestions.is_empty() {
                        suggestions[0].clone()
                    } else {
                        Suggestion::new(
                            term_list1[1].as_str(),
                            edit_distance_max + 1,
                            10 / (10i64).pow(self.string_strategy.len(&term_list1[i]) as u32),
                        )
                    };

                    //if (suggestions_combi[0].distance + 1 < DamerauLevenshteinDistance(term_list1[i - 1] + " " + term_list1[i], best1.term + " " + best2.term))
                    let distance1 = best1.distance + best2.distance;

                    if (distance1 >= 0)
                        && (suggestions_combi[0].distance + 1 < distance1
                            || (suggestions_combi[0].distance + 1 == distance1
                                && (suggestions_combi[0].count
                                    > best1.count / self.corpus_word_count * best2.count)))
                    {
                        suggestions_combi[0].distance += 1;
                        let last_i = suggestion_parts.len() - 1;
                        suggestion_parts[last_i] = suggestions_combi[0].clone();
                        last_combi = true;
                        continue;
                    }
                }
            }
            last_combi = false;

            //alway split terms without suggestion / never split terms with suggestion ed=0 / never split single char terms
            if !suggestions.is_empty()
                && ((suggestions[0].distance == 0)
                    || (self.string_strategy.len(&term_list1[i]) == 1))
            {
                //choose best suggestion
                suggestion_parts.push(suggestions[0].clone());
            } else {
                let mut suggestion_split_best = if !suggestions.is_empty() {
                    //add original term
                    suggestions[0].clone()
                } else {
                    //if no perfect suggestion, split word into pairs
                    Suggestion::empty()
                };

                let term_length = self.string_strategy.len(&term_list1[i]);

                if term_length > 1 {
                    for j in 1..term_length {
                        let part1 = self.string_strategy.slice(&term_list1[i], 0, j);
                        let part2 = self.string_strategy.slice(&term_list1[i], j, term_length);

                        let mut suggestion_split = Suggestion::empty();

                        let suggestions1 = self.lookup(&part1, Verbosity::Top, edit_distance_max);

                        if !suggestions1.is_empty() {
                            let suggestions2 =
                                self.lookup(&part2, Verbosity::Top, edit_distance_max);

                            if !suggestions2.is_empty() {
                                //select best suggestion for split pair
                                suggestion_split.term =
                                    format!("{} {}", suggestions1[0].term, suggestions2[0].term);

                                let distance2 = quick_compare(
                                    &term_list1[i],
                                    &format!("{} {}", suggestions1[0].term, suggestions2[0].term),
                                    edit_distance_max as u8,
                                ).map(|v| v as i64)
                                    .unwrap_or_else(|| (edit_distance_max + 1) as i64);

                                if suggestion_split_best.term != "" {
                                    if distance2 > suggestion_split_best.distance {
                                        continue;
                                    }
                                    if distance2 < suggestion_split_best.distance {
                                        suggestion_split_best = Suggestion::empty();
                                    }
                                }
                                let count2: i64 = cmp::min(
                                    i64::MAX,
                                    ((suggestions1[0].count as f64)
                                        / (self.corpus_word_count as f64)
                                        * (suggestions2[0].count as f64))
                                        as i64,
                                );

                                suggestion_split.distance = distance2;
                                suggestion_split.count = count2;

                                //early termination of split
                                if suggestion_split_best.term == ""
                                    || suggestion_split.count > suggestion_split_best.count
                                {
                                    suggestion_split_best = suggestion_split.clone();
                                }
                            }
                        }
                    }

                    if suggestion_split_best.term != "" {
                        //select best suggestion for split pair
                        suggestion_parts.push(suggestion_split_best.clone());
                    } else {
                        let mut si = Suggestion::empty();
                        // NOTE: this effectively clamps si_count to a certain minimum value, which it can't go below
                        let si_count: f64 = 10f64
                            / ((10i64)
                                .saturating_pow(self.string_strategy.len(&term_list1[i]) as u32))
                                as f64;

                        si.term = term_list1[i].clone();
                        si.count = si_count as i64;
                        si.distance = edit_distance_max + 1;
                        suggestion_parts.push(si);
                    }
                } else {
                    let mut si = Suggestion::empty();
                    // NOTE: this effectively clamps si_count to a certain minimum value, which it can't go below
                    let si_count: f64 = 10f64
                        / ((10i64).saturating_pow(self.string_strategy.len(&term_list1[i]) as u32))
                            as f64;

                    si.term = term_list1[i].clone();
                    si.count = si_count as i64;
                    si.distance = edit_distance_max + 1;
                    suggestion_parts.push(si);
                }
            }
        }

        let mut s = "".to_string();
        for si in suggestion_parts {
            s.push_str(&si.term);
            s.push_str(" ");
        }

        s.trim().to_string()
    }

    /// Divides a string into words by inserting missing spaces at the appropriate positions
    ///
    ///
    /// # Arguments
    ///
    /// * `input` - The word being segmented.
    /// * `max_edit_distance` - The maximum edit distance between input and suggested words.
    pub fn word_segmentation(&self, input: &str, max_edit_distance: i64) -> Composition {
        let input = self.string_strategy.prepare(input);
        let asize = self.string_strategy.len(&input);

        let mut ci: usize = 0;
        let mut compositions: Vec<Composition> = vec![Composition::empty(); asize];

        for j in 0..asize {
            let imax = cmp::min(asize - j, self.max_length as usize);
            for i in 1..=imax {
                let top_prob_log: f64;

                let mut part = self.string_strategy.slice(&input, j, j + i);

                let mut sep_len = 0;
                let mut top_ed: i64 = 0;

                let first_char = self.string_strategy.at(&part, 0).unwrap();
                if first_char.is_whitespace() {
                    part = self.string_strategy.remove(&part, 0);
                } else {
                    sep_len = 1;
                }

                top_ed += part.len() as i64;

                part = part.replace(" ", "");

                top_ed -= part.len() as i64;

                let results = self.lookup(&part, Verbosity::Top, max_edit_distance);

                if !results.is_empty() && results[0].distance == 0 {
                    top_prob_log =
                        (results[0].count as f64 / self.corpus_word_count as f64).log10();
                } else {
                    top_ed += part.len() as i64;
                    top_prob_log = (10.0
                        / (self.corpus_word_count as f64 * 10.0f64.powf(part.len() as f64)))
                    .log10();
                }

                let di = (i + ci) % asize;
                // set values in first loop
                if j == 0 {
                    compositions[i - 1] = Composition {
                        segmented_string: part.to_owned(),
                        distance_sum: top_ed,
                        prob_log_sum: top_prob_log,
                    };
                } else if i as i64 == self.max_length
                    || (((compositions[ci].distance_sum + top_ed == compositions[di].distance_sum)
                        || (compositions[ci].distance_sum + sep_len + top_ed
                            == compositions[di].distance_sum))
                        && (compositions[di].prob_log_sum
                            < compositions[ci].prob_log_sum + top_prob_log))
                    || (compositions[ci].distance_sum + sep_len + top_ed
                        < compositions[di].distance_sum)
                {
                    compositions[di] = Composition {
                        segmented_string: format!("{} {}", compositions[ci].segmented_string, part),
                        distance_sum: compositions[ci].distance_sum + sep_len + top_ed,
                        prob_log_sum: compositions[ci].prob_log_sum + top_prob_log,
                    };
                }
            }
            if j != 0 {
                ci += 1;
            }
            ci = if ci == asize { 0 } else { ci };
        }
        compositions[ci].to_owned()
    }

    fn delete_in_suggestion_prefix(
        &self,
        delete: &str,
        delete_len: i64,
        suggestion: &str,
        suggestion_len: i64,
    ) -> bool {
        if delete_len == 0 {
            return true;
        }
        let suggestion_len = if self.prefix_length < suggestion_len {
            self.prefix_length
        } else {
            suggestion_len
        };
        let mut j = 0;
        for i in 0..delete_len {
            let del_char = self.string_strategy.at(delete, i as isize).unwrap();
            while j < suggestion_len
                && del_char != self.string_strategy.at(suggestion, j as isize).unwrap()
            {
                j += 1;
            }

            if j == suggestion_len {
                return false;
            }
        }
        true
    }

    pub fn using_with_dictionary(&mut self, map: HashMap<String, i64>) {
        let mut deletes = HashMap::new();
        let mut words = HashMap::new();
        for (word, freq) in map {
            self.create_dictionary_entry(word, freq, &mut deletes, &mut words);
        }

        self.deletes.using_dictionary(deletes);
        self.words = words;
    }

    fn create_dictionary_entry(
        &mut self,
        key: String,
        count: i64,
        deletes: &mut HashMap<String, Vec<String>>,
        words: &mut HashMap<String, i64>,
    ) {
        if count < self.count_threshold {
            return;
        }

        let key_clone = key.clone();

        match words.get(key.as_str()) {
            Some(i) => {
                let updated_count = if i64::MAX - i > count {
                    i + count
                } else {
                    i64::MAX
                };
                words.insert(key_clone, updated_count);
                return;
            }
            None => {
                words.insert(key_clone, count);
            }
        }

        let key_len = self.string_strategy.len(key.as_ref());

        if key_len as i64 > self.max_length {
            self.max_length = key_len as i64;
        }

        let edits = self.edits_prefix(key.as_ref());

        for delete in edits {
            deletes
                .entry(delete)
                .and_modify(|e| e.push(key.clone()))
                .or_insert_with(|| vec![key.clone()]);
        }
    }

    fn edits_prefix(&self, key: &str) -> HashSet<String> {
        let mut hash_set = HashSet::new();

        let key_len = self.string_strategy.len(key) as i64;

        if key_len <= self.max_dictionary_edit_distance {
            hash_set.insert("".to_string());
        }

        if key_len > self.prefix_length {
            let shortened_key = self
                .string_strategy
                .slice(key, 0, self.prefix_length as usize);
            hash_set.insert(shortened_key.clone());
            self.edits(&shortened_key, 0, &mut hash_set);
        } else {
            hash_set.insert(key.to_string());
            self.edits(key, 0, &mut hash_set);
        };

        hash_set
    }

    fn edits(&self, word: &str, edit_distance: i64, delete_words: &mut HashSet<String>) {
        let edit_distance = edit_distance + 1;
        let word_len = self.string_strategy.len(word);

        if word_len > 1 {
            for i in 0..word_len {
                let delete = self.string_strategy.remove(word, i);

                if !delete_words.contains(&delete) {
                    delete_words.insert(delete.clone());

                    if edit_distance < self.max_dictionary_edit_distance {
                        self.edits(&delete, edit_distance, delete_words);
                    }
                }
            }
        }
    }

    fn has_different_suffix(
        &self,
        max_edit_distance: i64,
        input: &str,
        input_len: i64,
        candidate_len: i64,
        suggestion: &str,
        suggestion_len: i64,
    ) -> bool {
        // handles the shortcircuit of min_distance
        // assignment when first boolean expression
        // evaluates to false
        let min = if self.prefix_length - max_edit_distance == candidate_len {
            cmp::min(input_len, suggestion_len) - self.prefix_length
        } else {
            0
        };

        (self.prefix_length - max_edit_distance == candidate_len)
            && (((min - self.prefix_length) > 1)
                && (self
                    .string_strategy
                    .suffix(input, (input_len + 1 - min) as usize)
                    != self
                        .string_strategy
                        .suffix(suggestion, (suggestion_len + 1 - min) as usize)))
            || ((min > 0)
                && (self.string_strategy.at(input, (input_len - min) as isize)
                    != self
                        .string_strategy
                        .at(suggestion, (suggestion_len - min) as isize))
                && ((self
                    .string_strategy
                    .at(input, (input_len - min - 1) as isize)
                    != self
                        .string_strategy
                        .at(suggestion, (suggestion_len - min) as isize))
                    || (self.string_strategy.at(input, (input_len - min) as isize)
                        != self
                            .string_strategy
                            .at(suggestion, (suggestion_len - min - 1) as isize))))
    }

    fn parse_words(&self, text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .map(|s| s.to_string())
            .collect()
    }
}

#[inline]
fn quick_compare(s1: &str, s2: &str, max_distance: u8) -> Option<u8> {
    use triple_accel::levenshtein;

    let s1 = s1.as_bytes();
    let s2 = s2.as_bytes();

    let max_len = cmp::max(s1.len(), s2.len());
    let min_len = cmp::max(s1.len(), s2.len());

    // optimisation, if we know the size delta is greater than our max edit distance there's
    // no point doing the costly compute.
    if (max_len - min_len) > max_distance as usize {
        return None
    }

    levenshtein::levenshtein_simd_k_with_opts(
        s1,
        s2,
        max_distance as u32,
        false,
        levenshtein::RDAMERAU_COSTS,
    ).map(|(v, _)| v as u8)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::string_strategy::UnicodeStringStrategy;

    #[test]
    fn test_lookup_compound() {
        let edit_distance_max = 2;
        let mut sym_spell = SymSpell::<UnicodeStringStrategy>::default();
        sym_spell.using_with_dictionary(get_frequencies());

        let typo = "whereis th elove";
        let correction = "whereas the love";
        let results = sym_spell.lookup_compound(typo, edit_distance_max);
        assert_eq!(correction, results);

        let typo = "the bigjest playrs";
        let correction = "the biggest players";
        let results = sym_spell.lookup_compound(typo, edit_distance_max);
        assert_eq!(correction, results);

        let typo = "Can yu readthis";
        let correction = "can you read this";
        let results = sym_spell.lookup_compound(typo, edit_distance_max);
        assert_eq!(correction, results);

        let typo = "whereis th elove hehad dated forImuch of thepast who couqdn'tread in sixthgrade and ins pired him";
        let correction = "whereas the love he had dated for much of the past who couldn't read in sixth grade and inspired him";
        let results = sym_spell.lookup_compound(typo, edit_distance_max);
        assert_eq!(correction, results);

        let typo = "in te dhird qarter oflast jear he hadlearned ofca sekretplan";
        let correction = "in the third quarter of last year he had learned of a secret plan";
        let results = sym_spell.lookup_compound(typo, edit_distance_max);
        assert_eq!(correction, results);

        let typo = "the bigjest playrs in te strogsommer film slatew ith plety of funn";
        let correction = "the biggest players in the strong summer film slate with plenty of fun";
        let results = sym_spell.lookup_compound(typo, edit_distance_max);
        assert_eq!(correction, results);

        let typo = "Can yu readthis messa ge despite thehorible sppelingmsitakes";
        let correction = "can you read this message despite the horrible spelling mistakes";
        let results = sym_spell.lookup_compound(typo, edit_distance_max);
        assert_eq!(correction, results);
    }

    #[test]
    fn test_word_segmentation() {
        let edit_distance_max = 2;
        let mut sym_spell = SymSpell::<UnicodeStringStrategy>::default();
        sym_spell.using_with_dictionary(get_frequencies());

        let typo = "thequickbrownfoxjumpsoverthelazydog";
        let correction = "the quick brown fox jumps over the lazy dog";
        let result = sym_spell.word_segmentation(typo, edit_distance_max);
        assert_eq!(correction, result.segmented_string);

        let typo = "itwasabrightcolddayinaprilandtheclockswerestrikingthirteen";
        let correction = "it was a bright cold day in april and the clocks were striking thirteen";
        let result = sym_spell.word_segmentation(typo, edit_distance_max);
        assert_eq!(correction, result.segmented_string);

        let typo =
            "itwasthebestoftimesitwastheworstoftimesitwastheageofwisdomitwastheageoffoolishness";
        let correction = "it was the best of times it was the worst of times it was the age of wisdom it was the age of foolishness";
        let result = sym_spell.word_segmentation(typo, edit_distance_max);
        assert_eq!(correction, result.segmented_string);
    }

    fn get_frequencies() -> HashMap<String, i64> {
        let mut sets = HashMap::new();
        let file = File::open("data/frequency_dictionary_en_82_765.txt").unwrap();
        let sr = BufReader::new(file);

        for (_, line) in sr.lines().enumerate() {
            let line_str = line.unwrap();
            let line_parts: Vec<&str> = line_str.split(" ").collect();
            if line_parts.len() >= 2 {
                let term = line_parts[0];
                let count = line_parts[1].parse::<i64>().unwrap();

                sets.insert(term.to_string(), count as i64);
            }
        }

        sets
    }
}
