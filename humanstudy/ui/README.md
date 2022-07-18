# Human subject study

We ran our studies through Human Intelligence Tasks (HITs) deployed on [Amazon Mechanical Turk](https://www.mturk.com/) (AMT).
We use [simple-amt](https://github.com/jcjohnson/simple-amt), a microframework for working with AMT. 
Our codebase was adapted and developed from [https://github.com/princetonvisualai/HIVE](https://github.com/princetonvisualai/HIVE).


## Our study UIs

- combined_8concept.html
- combined_16concept.html
- combined_32concept_sgroup1.html
- combined_32concept_sgroup2.html
- combined_example.html 


## Brief instructions on how to run user studies on AMT

Please check out the original [simple-amt](https://github.com/jcjohnson/simple-amt) repository for more information on how to run a HIT on AMT.

#### Launch HITs on AMT
```
python launch_hits.py \
--html_template=hit_templates/combined_8concept.html \
--hit_properties_file=hit_properties/properties.json \
--input_json_file=examples/input_example.txt \
--hit_ids_file=examples/hit_ids_example.txt --prod
```

#### Check HIT progress
```
python show_hit_progress.py \
--hit_ids_file=examples/hit_ids_example.txt --prod
```

#### Get results
```
python get_results.py \
  --hit_ids_file=examples/hit_ids_example.txt \
  --output_file=examples/results_example.txt \
  > examples/results_example.txt --prod
```

#### Approve work
```
python approve_hits.py \
--hit_ids_file=examples/hit_ids_example.txt --prod
```
