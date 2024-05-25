use std::collections::HashSet;
use std::io::Cursor;
use heed::EnvOpenOptions;
use milli::documents::{DocumentsBatchBuilder, DocumentsBatchReader};
use milli::{Criterion, DefaultSearchLogger, execute_search, filtered_universe, GeoSortStrategy, Object, SearchContext, TermsMatchingStrategy, TimeBudget};
use milli::update::{DocumentAdditionResult, IndexDocuments, IndexDocumentsConfig, IndexerConfig, Settings};
use pyo3::exceptions::PyRuntimeError;
use pyo3::prelude::*;


#[pyclass]
struct MilliEmbedded {
    index: milli::Index,
}

#[pymethods]
impl MilliEmbedded {
    #[new]
    fn new(index_path: &str, searchable_fields: Vec<String>, filterable_fields: HashSet<String>) -> PyResult<Self> {
        std::fs::create_dir_all(&index_path)
            .map_err(|e| PyRuntimeError::new_err(format!("Cannot create index path: {e}")))?;

        let options = EnvOpenOptions::new();
        let index = milli::Index::new(options, index_path)
            .map_err(|e| PyRuntimeError::new_err(format!("Cannot create index, {e}")))?;

        let mut wtxn = index.write_txn().unwrap();
        let config = IndexerConfig::default();
        let mut builder = Settings::new(&mut wtxn, &index, &config);
        builder.set_searchable_fields(searchable_fields);
        builder.set_filterable_fields(filterable_fields);
        builder.set_criteria(vec![
            Criterion::Words,
            Criterion::Typo,
            Criterion::Proximity,
            Criterion::Attribute,
            Criterion::Sort,
            Criterion::Exactness,
        ]);

        builder.execute(|_| (), || false).map_err(|e| PyRuntimeError::new_err(format!("Cannot execute index builder, {e}")))?;

        wtxn.commit().map_err(|e| PyRuntimeError::new_err(format!("Cannot commit transaction, {e}")))?;

        Ok(Self { index })
    }

    fn mutate(&self, py: Python, add_jsonl: String, remove_ids: Vec<String>) -> PyResult<(u64, u64)> {
        let mut build_res = DocumentAdditionResult {
            indexed_documents: 0,
            number_of_documents: 0,
        };

        py.allow_threads(|| {
            let mut wtxn = self.index.write_txn().map_err(|e| PyRuntimeError::new_err(format!("Cannot create write transaction, {e}")))?;

            let config = IndexerConfig::default();

            if !remove_ids.is_empty() {
                let indexing_config = IndexDocumentsConfig::default();
                let builder =
                    IndexDocuments::new(&mut wtxn, &self.index, &config, indexing_config, |_| (), || false)
                        .map_err(|e| PyRuntimeError::new_err(format!("Cannot create index documents builder, {e}")))?;

                let (builder, user_error) = builder.remove_documents(remove_ids).map_err(|e| PyRuntimeError::new_err(format!("Cannot delete documents, {e}")))?;
                user_error.map_err(|e| PyRuntimeError::new_err(format!("User error, {e}")))?;
                build_res = builder.execute().map_err(|e| PyRuntimeError::new_err(format!("Cannot execute builder, {e}")))?;
            }

            if !add_jsonl.is_empty() {
                let indexing_config = IndexDocumentsConfig::default();
                let builder =
                    IndexDocuments::new(&mut wtxn, &self.index, &config, indexing_config, |_| (), || false)
                        .map_err(|e| PyRuntimeError::new_err(format!("Cannot create index documents builder, {e}")))?;

                let mut sources = DocumentsBatchBuilder::new(Vec::new());

                for result in serde_json::Deserializer::from_str(&add_jsonl).into_iter::<Object>() {
                    let object = result.map_err(|e| PyRuntimeError::new_err(format!("Cannot deserialize object, {e}")))?;
                    sources.append_json_object(&object).map_err(|e| PyRuntimeError::new_err(format!("Cannot deserialize object, {e}")))?;
                }

                let sources = sources.into_inner().map_err(|e| PyRuntimeError::new_err(format!("Cannot get sources, {e}")))?;

                let documents = DocumentsBatchReader::from_reader(Cursor::new(sources))
                    .map_err(|e| PyRuntimeError::new_err(format!("Cannot create documents batch reader, {e}")))?;

                let (builder, user_error) = builder.add_documents(documents).map_err(|e| PyRuntimeError::new_err(format!("Cannot add documents, {e}")))?;
                user_error.map_err(|e| PyRuntimeError::new_err(format!("User error, {e}")))?;
                build_res = builder.execute().map_err(|e| PyRuntimeError::new_err(format!("Cannot execute builder, {e}")))?;
            }


            wtxn.commit().map_err(|e| PyRuntimeError::new_err(format!("Cannot commit transaction, {e}")))
        })?;

        Ok((build_res.indexed_documents, build_res.number_of_documents))
    }

    fn search(&self, py: Python, query: String, return_fields: HashSet<String>) -> PyResult<String> {
        py.allow_threads(|| {
            let txn = self.index.read_txn().map_err(|e| PyRuntimeError::new_err(format!("{}", e)))?;
            let mut ctx = SearchContext::new(&self.index, &txn).map_err(|e| PyRuntimeError::new_err(format!("{}", e)))?;
            let universe = filtered_universe(&ctx, &None).map_err(|e| PyRuntimeError::new_err(format!("{}", e)))?;
            let docs = execute_search(
                &mut ctx,
                (!query.trim().is_empty()).then(|| query.trim()),
                TermsMatchingStrategy::Last,
                milli::score_details::ScoringStrategy::Skip,
                false,
                universe,
                &None,
                GeoSortStrategy::default(),
                0,
                20,
                None,
                &mut DefaultSearchLogger,
                &mut DefaultSearchLogger,
                TimeBudget::max(),
            ).map_err(|e| PyRuntimeError::new_err(format!("{e}")))?;

            let documents = self.index
                .documents(&txn, docs.documents_ids.iter().copied())
                .unwrap()
                .into_iter()
                .map(|(_id, obkv)| {
                    let mut object = serde_json::Map::default();
                    for (fid, fid_name) in self.index.fields_ids_map(&txn).unwrap().iter() {
                        if !return_fields.contains(fid_name) {
                            continue;
                        }
                        let value = obkv.get(fid).unwrap();
                        let value: serde_json::Value = serde_json::from_slice(value).unwrap();
                        object.insert(fid_name.to_owned(), value);
                    }
                    object
                })
                .collect::<serde_json::Value>();
            Ok(format!("{documents}"))
        })
    }
}

#[pymodule]
fn milli_py(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<MilliEmbedded>()?;
    Ok(())
}
