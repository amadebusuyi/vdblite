//! Typed document metadata and its sortable index encoding.
//!
//! The metadata index maps `field ∅ tag value` to a bitmap of chunk ids. Keys
//! sort so that a numeric range is one contiguous key range: all numbers are
//! indexed as f64 under one tag, encoded so byte order equals numeric order.

use serde::{Deserialize, Serialize};

/// One metadata value.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum MetaValue {
    Null,
    Bool(bool),
    Int(i64),
    Float(f64),
    Str(String),
    /// A list of scalars; each element is indexed, so an equality filter
    /// matches when any element matches.
    List(Vec<MetaValue>),
}

impl MetaValue {
    /// The value as a number, if it is one.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            MetaValue::Int(i) => Some(*i as f64),
            MetaValue::Float(f) => Some(*f),
            _ => None,
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            MetaValue::Str(s) => Some(s),
            _ => None,
        }
    }

    /// Scalars this value contributes to the index (list elements, or itself).
    pub fn scalars(&self) -> Vec<&MetaValue> {
        match self {
            MetaValue::List(items) => items
                .iter()
                .filter(|v| !matches!(v, MetaValue::List(_)))
                .collect(),
            v => vec![v],
        }
    }

    /// Equality as the index sees it: numbers compare by value across Int
    /// and Float.
    pub fn index_eq(&self, other: &MetaValue) -> bool {
        match (self.as_f64(), other.as_f64()) {
            (Some(a), Some(b)) => a == b,
            _ => self == other,
        }
    }
}

macro_rules! from_impl {
    ($($t:ty => $v:ident),*) => {$(
        impl From<$t> for MetaValue {
            fn from(x: $t) -> Self { MetaValue::$v(x.into()) }
        }
    )*};
}
from_impl!(bool => Bool, i64 => Int, i32 => Int, u32 => Int, f64 => Float, f32 => Float, String => Str, &str => Str);

impl<T: Into<MetaValue>> From<Vec<T>> for MetaValue {
    fn from(v: Vec<T>) -> Self {
        MetaValue::List(v.into_iter().map(Into::into).collect())
    }
}

const TAG_NULL: u8 = 0;
const TAG_BOOL: u8 = 1;
const TAG_NUM: u8 = 2;
const TAG_STR: u8 = 3;

/// Separates the field name from the value in index keys. Field names may
/// not contain it.
pub const FIELD_END: u8 = 0;

/// The key under which every live chunk id is kept (for `Not` filters).
pub const ALL_KEY: &[u8] = b"\0all";

/// Whether a field name can be indexed.
pub fn valid_field(field: &str) -> bool {
    !field.is_empty() && !field.as_bytes().contains(&FIELD_END)
}

fn field_prefix(field: &str) -> Vec<u8> {
    let mut k = Vec::with_capacity(field.len() + 10);
    k.extend_from_slice(field.as_bytes());
    k.push(FIELD_END);
    k
}

/// Order-preserving encoding of an f64: byte order equals numeric order,
/// with -0.0 folded into 0.0.
pub fn encode_f64(x: f64) -> [u8; 8] {
    let x = if x == 0.0 { 0.0 } else { x };
    let bits = x.to_bits();
    let ordered = if bits >> 63 == 1 {
        !bits
    } else {
        bits | (1 << 63)
    };
    ordered.to_be_bytes()
}

/// The index key of one scalar value of `field`; `None` for lists.
pub fn index_key(field: &str, value: &MetaValue) -> Option<Vec<u8>> {
    let mut k = field_prefix(field);
    match value {
        MetaValue::Null => k.push(TAG_NULL),
        MetaValue::Bool(b) => {
            k.push(TAG_BOOL);
            k.push(u8::from(*b));
        }
        MetaValue::Int(_) | MetaValue::Float(_) => {
            k.push(TAG_NUM);
            k.extend_from_slice(&encode_f64(value.as_f64().expect("numeric")));
        }
        MetaValue::Str(s) => {
            k.push(TAG_STR);
            k.extend_from_slice(s.as_bytes());
        }
        MetaValue::List(_) => return None,
    }
    Some(k)
}

/// Every index key a field's value occupies (one per scalar).
pub fn index_keys(field: &str, value: &MetaValue) -> Vec<Vec<u8>> {
    value
        .scalars()
        .into_iter()
        .filter_map(|v| index_key(field, v))
        .collect()
}

/// Key bounds covering every value of `field`: `[start, end)`.
pub fn field_bounds(field: &str) -> (Vec<u8>, Vec<u8>) {
    let start = field_prefix(field);
    let mut end = field.as_bytes().to_vec();
    end.push(FIELD_END + 1);
    (start, end)
}

/// Key of the numeric value `x` of `field`, for building range bounds.
pub fn number_key(field: &str, x: f64) -> Vec<u8> {
    let mut k = field_prefix(field);
    k.push(TAG_NUM);
    k.extend_from_slice(&encode_f64(x));
    k
}

/// Key bounds covering every numeric value of `field`: `[start, end)`.
pub fn number_bounds(field: &str) -> (Vec<u8>, Vec<u8>) {
    let mut start = field_prefix(field);
    start.push(TAG_NUM);
    let mut end = field_prefix(field);
    end.push(TAG_NUM + 1);
    (start, end)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn number_encoding_sorts_numerically() {
        let mut xs = [
            -1e300,
            -5.5,
            -1.0,
            -0.0,
            0.0,
            1e-9,
            1.0,
            2.0,
            1e12,
            f64::INFINITY,
        ];
        let mut keys: Vec<[u8; 8]> = xs.iter().map(|&x| encode_f64(x)).collect();
        keys.sort();
        xs.sort_by(f64::total_cmp);
        let decoded_order: Vec<[u8; 8]> = xs.iter().map(|&x| encode_f64(x)).collect();
        assert_eq!(keys, decoded_order);
        assert_eq!(encode_f64(-0.0), encode_f64(0.0));
    }

    #[test]
    fn ints_and_floats_share_keys() {
        assert_eq!(
            index_key("n", &MetaValue::Int(3)),
            index_key("n", &MetaValue::Float(3.0))
        );
        assert!(MetaValue::Int(3).index_eq(&MetaValue::Float(3.0)));
    }

    #[test]
    fn lists_index_each_element() {
        let v = MetaValue::from(vec!["a", "b"]);
        assert_eq!(index_keys("tags", &v).len(), 2);
    }

    #[test]
    fn field_bounds_cover_only_that_field() {
        let (lo, hi) = field_bounds("lang");
        let en = index_key("lang", &"en".into()).unwrap();
        let other = index_key("language", &"en".into()).unwrap();
        assert!(lo <= en && en < hi);
        assert!(!(lo <= other && other < hi));
    }
}
