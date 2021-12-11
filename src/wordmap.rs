use std::fmt::{Debug, Display, Formatter};
use std::hash::{Hash, Hasher};
use std::iter::FromIterator;
use std::ops::Deref;

use hashbrown::{HashMap, HashSet};

#[derive(Clone)]
pub(crate) struct WordRef(*const Word);

impl WordRef {
    #[inline]
    pub(crate) fn len(&self) -> usize {
        let ref_ = unsafe { &*self.0 };
        ref_.len()
    }

    #[inline]
    pub(crate) fn as_str(&self) -> &str {
        let ref_ = unsafe { &*self.0 };
        ref_.as_str()
    }
}

impl Debug for WordRef {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "WordRef({:?})", self.as_str())
    }
}

impl Deref for WordRef {
    type Target = Word;

    fn deref(&self) -> &Self::Target {
        unsafe { &*self.0 }
    }
}

impl Display for WordRef {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl PartialEq<Self> for WordRef {
    fn eq(&self, other: &Self) -> bool {
        (self.0).eq(&other.0)
    }
}

impl PartialEq<[u8]> for WordRef {
    fn eq(&self, other: &[u8]) -> bool {
        self.as_str().as_bytes().eq(other)
    }
}

impl PartialEq<str> for WordRef {
    fn eq(&self, other: &str) -> bool {
       self.as_str().eq(other)
    }
}

impl Eq for WordRef {}


#[derive(Clone, Default)]
pub(crate) struct Word(Box<[u8]>);

impl Word {
    #[inline]
    pub(crate) fn len(&self) -> usize {
        self.0.len()
    }

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
    word_references: HashSet<Word>,
}

impl Debug for WordMap {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.data)
    }
}

impl WordMap {
    pub(crate) fn using_dictionary(&mut self, mut dictionary: HashMap<String, Vec<String>>) {
        let ref_words = {
            let mut ref_words = HashSet::new();
            for words in dictionary.values() {
                for word in words {
                    ref_words.insert(Word::from(word.clone()));
                }
            }

            ref_words.shrink_to_fit();

            ref_words
        };

        let mut dict = HashMap::from_iter(
            dictionary.drain()
                .map(|(k, v)| {
                    let v: Vec<_> = v.into_iter()
                        .map(|w| {
                            let ref_word = ref_words.get(&Word::from(w)).unwrap();
                            let ptr = ref_word as *const Word;

                            WordRef(ptr)
                        })
                        .collect();

                    (self.hash_string(&k), v.into_boxed_slice())
                })
        );

        dict.shrink_to_fit();

        self.data = dict;
        self.word_references = ref_words;
    }

    pub(crate) fn get(&self, word: &str) -> Option<&[WordRef]> {
        self.data.get(&self.hash_string(word)).map(|v| v.as_ref())
    }

    fn hash_string(&self, s: &str) -> u64 {
        let mut hasher = ahash::AHasher::default();
        s.hash(&mut hasher);

        hasher.finish()
    }
}