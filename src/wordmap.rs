use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::iter::FromIterator;
use std::ops::Deref;

use hashbrown::HashMap;

/// A 32 bit sized pointer to a given word.
#[derive(Copy, Clone)]
pub(crate) struct WordRef(u32);

impl Debug for WordRef {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "WordRef(index={})", self.0)
    }
}

#[derive(Clone, Default)]
pub(crate) struct Word(Box<[u8]>);

impl Word {
    #[inline]
    pub(crate) fn as_str(&self) -> &str {
        unsafe { std::str::from_utf8_unchecked(self.0.as_ref()) }
    }
}

impl Debug for Word {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl Deref for Word {
    type Target = Box<[u8]>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl From<String> for Word {
    fn from(s: String) -> Self {
        Word(s.into_boxed_str().into_boxed_bytes())
    }
}

impl From<&str> for Word {
    fn from(s: &str) -> Self {
        Word(s.to_owned().into_boxed_str().into_boxed_bytes())
    }
}

impl Display for Word {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl PartialEq<Self> for Word {
    fn eq(&self, other: &Self) -> bool {
        self.0.eq(&other.0)
    }
}

impl PartialEq<[u8]> for Word {
    fn eq(&self, other: &[u8]) -> bool {
        (*self.0).eq(other)
    }
}

impl PartialEq<str> for Word {
    fn eq(&self, other: &str) -> bool {
        (*self.0).eq(other.as_bytes())
    }
}

impl Eq for Word {}

impl Hash for Word {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.0.hash(state)
    }
}

#[derive(Default)]
pub(crate) struct WordMap {
    data: HashMap<u64, Box<[WordRef]>>,
    word_references: Box<[Word]>,
}

impl Debug for WordMap {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.data)
    }
}

impl WordMap {
    pub(crate) fn using_dictionary(&mut self, mut dictionary: HashMap<String, Vec<String>>) {
        let (ref_words, lookup) = {
            let mut lookup_index: HashMap<String, u32> = HashMap::new();
            let mut ref_words = Vec::new();
            for words in dictionary.values() {
                for word in words {
                    if !lookup_index.contains_key(word) {
                        ref_words.push(Word::from(word.clone()));
                        lookup_index.insert(word.clone(), (ref_words.len() - 1) as u32);
                    }
                }
            }

            let slice = Vec::from_iter(ref_words.into_iter()).into_boxed_slice();

            (slice, lookup_index)
        };

        let mut dict = HashMap::from_iter(dictionary.drain().map(|(k, v)| {
            let v: Vec<_> = v
                .into_iter()
                .map(|w| {
                    let ptr = lookup.get(&w).unwrap();
                    WordRef(*ptr)
                })
                .collect();

            (self.hash_string(&k), v.into_boxed_slice())
        }));

        dict.shrink_to_fit();

        self.data = dict;
        self.word_references = ref_words;
    }

    #[inline]
    pub(crate) fn word_at(&self, word_ref: &WordRef) -> &Word {
        unsafe { self.word_references.get_unchecked(word_ref.0 as usize) }
    }

    #[inline]
    pub(crate) fn get(&self, word: &str) -> Option<&[WordRef]> {
        self.data.get(&self.hash_string(word)).map(|v| v.as_ref())
    }

    fn hash_string(&self, s: &str) -> u64 {
        let mut hasher = ahash::AHasher::default();
        s.hash(&mut hasher);

        hasher.finish()
    }
}
