NewsScope Benchmark Dataset
===========================

EXTRACTION:
  unzip benchmark.zip -d data/benchmark/

This will create:
  data/benchmark/train.jsonl (315 articles)
  data/benchmark/test_indomain.jsonl (80 articles)
  data/benchmark/test_oos.jsonl (60 articles)

TOTAL: 455 articles (395 in-domain + 60 out-of-source)

NOTE: Article text is NOT included due to copyright.
Use the URLs in each entry to fetch article content if needed.

FORMAT (each line is JSON):
{
  "article_id": "unique_id",
  "url": "https://...",
  "domain": "politics|health|science_env|business",
  "source": "Source Name",
  "annotation": {...}
}

LINKS:
  GitHub: https://github.com/nidhip1611/NewsScope
  Model: https://huggingface.co/nidhipandya/NewsScope-lora
