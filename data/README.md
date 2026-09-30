# data/

Put the PDFs you want to query here, e.g.

```
data/
├── book1.pdf
├── book2.pdf
└── book3.pdf
```

Every `*.pdf` in this folder is loaded, chunked, embedded and stored in a local
FAISS index (`.index/`) the first time you run the pipeline.

**These files are git-ignored on purpose.** Textbooks and papers are usually
copyrighted, and redistributing them in a public repo is not okay. Tell readers
what kind of documents you used (for example, "three introductory ML textbooks")
and let them bring their own.

After adding or removing PDFs, rebuild the index:

```bash
python -m crag "any question" --rebuild-index
```
