# UrbanSensing-Benchmark

## Task

| cate        | indicator                   | cities         | case |
| ----------- | --------------------------- | -------------- | ---- |
| economic    | GDP                         | 13 cities      | 1000 |
|             | house price                 | us3 uk3 China2 | 777  |
|             | population                  | 13 cities      | 1000 |
| education   | bachelor ratio              | us 3 cities    |      |
| crime       | violent                     | us 3 cities    | 396  |
| transport   | drive                       | us 3 cities    | 500  |
|             | public                      | us 3 cities    | 500  |
| health      | mental health               | us 3 cities    | 500  |
|             | life expectancy             | uk 3 cities    | 193  |
|             | accessibility to healthcare | 13 cities      | 1000 |
| environment | carbon                      | 13 cities      | 1000 |
|             | building height             | 13 cities      | 1000 |

## Model

| method | model name                             |
| ------ | -------------------------------------- |
| api    | gpt-4.1-mini                           |
|        | gpt-4.1-nano                           |
|        | gemma-3-4b                             |
|        | gemma-3-12b                            |
|        | gemma-3-27b                            |
|        | Llama-4-Maverick-17B-128E-Instruct-FP8 |
|        | Llama-4-Scout-17B-16E-Instruct         |
|        | gemini-2.0-flash                       |
| local  | Qwen2.5-vl-3b                          |
|        | Qwen2.5-vl-7b                          |

## Evaluation for simple and normalized

You can use examples/run_eval_city.sh to evaluate. Prompt type can choose `simple` or `normalized`.

### Economic

```python
# GDP
python -m evaluate.global.global_indicator --city_name="all" --mode="eval" --model_name="gpt-4o" --prompt_type="simple" --num_process=10 --task_name="gdp"
python -m evaluate.global.metrics --city_name="all" --model_name="gpt-4o" --prompt_type="simple" --task_name="gdp"

# house price
python -m evaluate.house_price.house_price_us --city_name="all" --mode="eval" --model_name="gpt-4o" --prompt_type="simple" --num_process=10
python -m evaluate.house_price.metrics --city_name="all" --model_name="gpt-4o" --prompt_type="simple" 

# population
python -m evaluate.global.global_indicator --city_name="all" --mode="eval" --model_name="gpt-4o" --prompt_type="simple" --num_process=10 --task_name="pop"
python -m evaluate.global.metrics --city_name="all" --model_name="gpt-4o" --prompt_type="simple" --task_name="pop"

```

### Education

```python

```

### Crime

```python
# violent
python -m evaluate.crime.crime_us --city_name="US" --mode="eval" --model_name="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8" --prompt_type="simple" --task_name="violent"
python -m evaluate.crime.metrics --city_name="US" --model_name="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8" --prompt_type="simple" --task_name="violent"
```

### Transport

```python
# drive
python -m evaluate.transport.transport_us --city_name="US" --mode="eval" --model_name="gpt-4o" --prompt_type="simple" --task_name="drive"
python -m evaluate.transport.metrics --city_name="US" --model_name="gpt-4o" --prompt_type="simple"  --task_name="drive"

# public
python -m evaluate.transport.transport_us --city_name="US" --mode="eval" --model_name="gpt-4o" --prompt_type="simple" --task_name="public"
python -m evaluate.transport.metrics --city_name="US" --model_name="gpt-4o" --prompt_type="simple"  --task_name="public"
```

### Health

```python
# mental health
python -m evaluate.health.health_us --city_name="US" --mode="eval" --model_name="google/gemma-3-4b-it" --prompt_type="simple" --task_name="mental"
python -m evaluate.health.metrics --city_name="US" --model_name="google/gemma-3-4b-it" --prompt_type="simple" --task_name="mental"

# life expectancy
python -m evaluate.life_exp.life_exp_uk --city_name="UK" --mode="eval" --model_name="google/gemma-3-4b-it" --prompt_type="simple"
python -m evaluate.life_exp.metrics --city_name="UK" --model_name="google/gemma-3-4b-it" --prompt_type="simple"

# accessibility to health
python -m evaluate.global.global_indicator --city_name="all" --mode="eval" --model_name="gpt-4o" --prompt_type="simple" --num_process=10 --task_name="acc2health"
python -m evaluate.global.metrics --city_name="all" --model_name="gpt-4o" --prompt_type="simple" --task_name="acc2health"
```

### Environment

```python
# carbon
python -m evaluate.global.global_indicator --city_name="all" --mode="eval" --model_name="gpt-4o" --prompt_type="simple" --num_process=10 --task_name="carbon"
python -m evaluate.global.metrics --city_name="all" --model_name="gpt-4o" --prompt_type="simple" --task_name="carbon"

# building height
python -m evaluate.global.global_indicator --city_name="all" --mode="eval" --model_name="gpt-4o" --prompt_type="simple" --num_process=10 --task_name="build_height"
python -m evaluate.global.metrics --city_name="all" --model_name="gpt-4o" --prompt_type="simple" --task_name="build_height"
```

## Evaluation for feature

```python
# extract feature from LLM
python -m evaluate.feature.extract_feature
# extract answer from LLM
python extract_api.py
# align feature and reference
python feature.py
# LASSO regression
python regression.py
```
