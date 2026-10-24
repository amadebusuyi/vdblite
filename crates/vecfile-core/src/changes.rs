//! The change log: how processes reading a file follow its writer.
//!
//! The writer keeps each collection's vector index in memory and changes it
//! after every commit. A process that opened the file read-only has its own
//! copy of that index, loaded at open. To stay current it replays the
//! writer's index changes, which the writer appends here in the same
//! transaction as the data they describe: entry `seq` of a collection is the
//! `seq`-th change since the collection was created, starting at 1.
//!
//! The log is trimmed to its newest [`KEEP`] entries. A reader that falls
//! further behind sees a gap and reloads the collection from the file.

use redb::{ReadableTable, Table};

use crate::Result;
use crate::records::{Change, decode, encode};

/// Entries kept per collection.
pub const KEEP: u64 = 1024;

/// Trimming runs once every this many appends.
const TRIM_EVERY: u64 = 64;

type ChangesTable<'t> = Table<'t, (u32, u64), &'static [u8]>;

/// The sequence number of a collection's newest entry, or 0 if it has none.
pub fn last_seq(table: &impl ReadableTable<(u32, u64), &'static [u8]>, coll: u32) -> Result<u64> {
    Ok(
        match table.range((coll, 0)..=(coll, u64::MAX))?.next_back() {
            Some(row) => row?.0.value().1,
            None => 0,
        },
    )
}

/// Appends a change and returns its sequence number.
pub fn append(table: &mut ChangesTable<'_>, coll: u32, change: &Change) -> Result<u64> {
    let seq = last_seq(table, coll)? + 1;
    table.insert((coll, seq), encode(change).as_slice())?;
    if seq.is_multiple_of(TRIM_EVERY) && seq > KEEP {
        table.retain_in((coll, 0)..(coll, seq - KEEP), |_, _| false)?;
    }
    Ok(seq)
}

/// The entries after `after`, oldest first, or `None` if some of them were
/// trimmed (the caller must reload instead).
pub fn read_after(
    table: &impl ReadableTable<(u32, u64), &'static [u8]>,
    coll: u32,
    after: u64,
) -> Result<Option<Vec<(u64, Change)>>> {
    let mut out = Vec::new();
    for row in table.range((coll, after.saturating_add(1))..=(coll, u64::MAX))? {
        let (k, v) = row?;
        let seq = k.value().1;
        if seq != after + 1 + out.len() as u64 {
            return Ok(None);
        }
        out.push((seq, decode(v.value(), "change")?));
    }
    Ok(Some(out))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tables::CHANGES;
    use redb::{Database, ReadableDatabase};

    fn seal(n: u64) -> Change {
        Change::Seal { segment: n }
    }

    #[test]
    fn appends_reads_and_trims() {
        let dir = tempfile::tempdir().unwrap();
        let db = Database::create(dir.path().join("c.redb")).unwrap();
        let tx = db.begin_write().unwrap();
        {
            let mut t = tx.open_table(CHANGES).unwrap();
            assert_eq!(last_seq(&t, 1).unwrap(), 0);
            for n in 1..=3 {
                assert_eq!(append(&mut t, 1, &seal(n)).unwrap(), n);
            }
            append(&mut t, 2, &seal(9)).unwrap();
            assert_eq!(read_after(&t, 1, 0).unwrap().unwrap().len(), 3);
            assert_eq!(
                read_after(&t, 1, 1).unwrap().unwrap(),
                vec![(2, seal(2)), (3, seal(3))]
            );
            assert_eq!(read_after(&t, 1, 3).unwrap().unwrap(), vec![]);
            assert_eq!(read_after(&t, 2, 0).unwrap().unwrap(), vec![(1, seal(9))]);

            for n in 4..=KEEP + 2 * TRIM_EVERY {
                append(&mut t, 1, &seal(n)).unwrap();
            }
            let last = last_seq(&t, 1).unwrap();
            assert_eq!(last, KEEP + 2 * TRIM_EVERY);
            // Old entries are gone, so a reader that far behind must reload.
            assert_eq!(read_after(&t, 1, 0).unwrap(), None);
            assert_eq!(read_after(&t, 1, 10).unwrap(), None);
            // Recent ones are all there.
            let recent = read_after(&t, 1, last - KEEP / 2).unwrap().unwrap();
            assert_eq!(recent.len() as u64, KEEP / 2);
            assert_eq!(recent.last().unwrap().1, seal(last));
            // Other collections are untouched.
            assert_eq!(read_after(&t, 2, 0).unwrap().unwrap().len(), 1);
        }
        tx.commit().unwrap();
        let rx = db.begin_read().unwrap();
        let t = rx.open_table(CHANGES).unwrap();
        let kept = t.range((1, 0)..=(1, u64::MAX)).unwrap().count() as u64;
        assert!((KEEP..=KEEP + TRIM_EVERY).contains(&kept), "{kept}");
    }
}
