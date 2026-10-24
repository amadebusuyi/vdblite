//! Metadata filters, evaluated against the inverted index for search and
//! directly against documents for scans.

use std::ops::{Bound, Not};

use redb::ReadableTable;
use roaring::RoaringBitmap;
use vecfile_core::codec::read_bitmap;
use vecfile_core::meta::{self, ALL_KEY, MetaValue};

use crate::doc::Meta;
use crate::error::Result;

/// A condition on document metadata.
#[derive(Clone, Debug, PartialEq)]
pub enum Filter {
    /// The field equals the value (any element, for list fields).
    Eq(String, MetaValue),
    /// The field equals any of the values.
    In(String, Vec<MetaValue>),
    /// The field is a number within the bounds.
    Range {
        field: String,
        lo: Bound<f64>,
        hi: Bound<f64>,
    },
    /// The field is present.
    Exists(String),
    And(Vec<Filter>),
    Or(Vec<Filter>),
    Not(Box<Filter>),
}

/// Starts a filter on a metadata field: `meta("lang").eq("en")`.
pub fn meta(field: impl Into<String>) -> Field {
    Field(field.into())
}

/// A metadata field, waiting for a condition.
#[derive(Clone, Debug)]
pub struct Field(String);

impl Field {
    pub fn eq(self, v: impl Into<MetaValue>) -> Filter {
        Filter::Eq(self.0, v.into())
    }

    /// Not equal (also true when the field is absent).
    pub fn ne(self, v: impl Into<MetaValue>) -> Filter {
        !self.eq(v)
    }

    pub fn is_in<V: Into<MetaValue>>(self, values: impl IntoIterator<Item = V>) -> Filter {
        Filter::In(self.0, values.into_iter().map(Into::into).collect())
    }

    pub fn gt(self, x: f64) -> Filter {
        Filter::Range {
            field: self.0,
            lo: Bound::Excluded(x),
            hi: Bound::Unbounded,
        }
    }

    pub fn gte(self, x: f64) -> Filter {
        Filter::Range {
            field: self.0,
            lo: Bound::Included(x),
            hi: Bound::Unbounded,
        }
    }

    pub fn lt(self, x: f64) -> Filter {
        Filter::Range {
            field: self.0,
            lo: Bound::Unbounded,
            hi: Bound::Excluded(x),
        }
    }

    pub fn lte(self, x: f64) -> Filter {
        Filter::Range {
            field: self.0,
            lo: Bound::Unbounded,
            hi: Bound::Included(x),
        }
    }

    /// `lo <= field <= hi`.
    pub fn between(self, lo: f64, hi: f64) -> Filter {
        Filter::Range {
            field: self.0,
            lo: Bound::Included(lo),
            hi: Bound::Included(hi),
        }
    }

    pub fn exists(self) -> Filter {
        Filter::Exists(self.0)
    }
}

impl Filter {
    pub fn and(self, other: Filter) -> Filter {
        match self {
            Filter::And(mut v) => {
                v.push(other);
                Filter::And(v)
            }
            f => Filter::And(vec![f, other]),
        }
    }

    pub fn or(self, other: Filter) -> Filter {
        match self {
            Filter::Or(mut v) => {
                v.push(other);
                Filter::Or(v)
            }
            f => Filter::Or(vec![f, other]),
        }
    }

    /// Whether a document's metadata satisfies the filter.
    pub fn matches(&self, m: &Meta) -> bool {
        match self {
            Filter::Eq(f, v) => m
                .get(f)
                .is_some_and(|x| any_scalar(x, |s| v.scalars().iter().any(|w| s.index_eq(w)))),
            Filter::In(f, vs) => m
                .get(f)
                .is_some_and(|x| any_scalar(x, |s| vs.iter().any(|w| s.index_eq(w)))),
            Filter::Range { field, lo, hi } => m.get(field).is_some_and(|x| {
                any_scalar(x, |s| s.as_f64().is_some_and(|n| in_bounds(n, *lo, *hi)))
            }),
            Filter::Exists(f) => m.contains_key(f),
            Filter::And(fs) => fs.iter().all(|f| f.matches(m)),
            Filter::Or(fs) => fs.iter().any(|f| f.matches(m)),
            Filter::Not(f) => !f.matches(m),
        }
    }

    /// The chunk ids whose documents satisfy the filter.
    pub(crate) fn eval(
        &self,
        index: &impl ReadableTable<(u32, &'static [u8]), &'static [u8]>,
        coll: u32,
    ) -> Result<RoaringBitmap> {
        let get = |key: &[u8]| -> Result<RoaringBitmap> {
            match index.get((coll, key))? {
                Some(v) => Ok(read_bitmap(v.value())?),
                None => Ok(RoaringBitmap::new()),
            }
        };
        let union_range = |lo: Bound<&[u8]>, hi: Bound<&[u8]>| -> Result<RoaringBitmap> {
            let mut out = RoaringBitmap::new();
            let lo = lo.map(|k| (coll, k));
            let hi = hi.map(|k| (coll, k));
            for row in index.range((lo, hi))? {
                let (_, v) = row?;
                out |= read_bitmap(v.value())?;
            }
            Ok(out)
        };
        Ok(match self {
            Filter::Eq(f, v) => {
                let mut out = RoaringBitmap::new();
                for k in meta::index_keys(f, v) {
                    out |= get(&k)?;
                }
                out
            }
            Filter::In(f, vs) => {
                let mut out = RoaringBitmap::new();
                for v in vs {
                    for k in meta::index_keys(f, v) {
                        out |= get(&k)?;
                    }
                }
                out
            }
            Filter::Range { field, lo, hi } => {
                let (start, end) = meta::number_bounds(field);
                let lo_key = lo.map(|x| meta::number_key(field, x));
                let hi_key = hi.map(|x| meta::number_key(field, x));
                let lo_b = match &lo_key {
                    Bound::Unbounded => Bound::Included(start.as_slice()),
                    b => b.as_ref().map(Vec::as_slice),
                };
                let hi_b = match &hi_key {
                    Bound::Unbounded => Bound::Excluded(end.as_slice()),
                    b => b.as_ref().map(Vec::as_slice),
                };
                if empty_range(lo, hi) {
                    RoaringBitmap::new()
                } else {
                    union_range(lo_b, hi_b)?
                }
            }
            Filter::Exists(f) => {
                let (start, end) = meta::field_bounds(f);
                union_range(Bound::Included(&start), Bound::Excluded(&end))?
            }
            Filter::And(fs) => {
                let mut it = fs.iter();
                let Some(first) = it.next() else {
                    return get(ALL_KEY);
                };
                let mut out = first.eval(index, coll)?;
                for f in it {
                    if out.is_empty() {
                        break;
                    }
                    out &= f.eval(index, coll)?;
                }
                out
            }
            Filter::Or(fs) => {
                let mut out = RoaringBitmap::new();
                for f in fs {
                    out |= f.eval(index, coll)?;
                }
                out
            }
            Filter::Not(f) => get(ALL_KEY)? - f.eval(index, coll)?,
        })
    }
}

impl Not for Filter {
    type Output = Filter;

    fn not(self) -> Filter {
        match self {
            Filter::Not(inner) => *inner,
            f => Filter::Not(Box::new(f)),
        }
    }
}

fn any_scalar(v: &MetaValue, pred: impl Fn(&MetaValue) -> bool) -> bool {
    v.scalars().into_iter().any(pred)
}

fn in_bounds(n: f64, lo: Bound<f64>, hi: Bound<f64>) -> bool {
    let above = match lo {
        Bound::Included(x) => n >= x,
        Bound::Excluded(x) => n > x,
        Bound::Unbounded => true,
    };
    let below = match hi {
        Bound::Included(x) => n <= x,
        Bound::Excluded(x) => n < x,
        Bound::Unbounded => true,
    };
    above && below
}

/// Bounds that admit nothing (which redb would reject as an inverted range).
fn empty_range(lo: &Bound<f64>, hi: &Bound<f64>) -> bool {
    match (lo, hi) {
        (Bound::Included(a), Bound::Included(b)) => a > b,
        (Bound::Included(a) | Bound::Excluded(a), Bound::Included(b) | Bound::Excluded(b)) => {
            a >= b
        }
        _ => false,
    }
}
