export CUDA_VISIBLE_DEVICES=2

export OPENAI_API_KEY="$OpenAI_API_KEY"
export OPENAI_API_BASE="https://api.openai.com/v1/chat/completions"

# CITIES=('Beijing' 'Shanghai' 'SanFrancisco' 'NewYork' 'Mumbai' 'Tokyo' 'London' 'Paris' 'Moscow' 'SaoPaulo' 'Nairobi' 'CapeTown' 'Sydney')
# for CITY in "${CITIES[@]}"; do
#     echo "Current city: $CITY"
#     python -m evaluate.global.global_indicator --city_name=$CITY --mode="gen"
# done
# GDP
# python -m evaluate.global.global_indicator --city_name="all" --mode="gen" 
# python -m evaluate.global.global_indicator --city_name="all" --mode="eval" --model_name="gpt-4.1-mini" --prompt_type="normalized" --num_process=10 --task_name="gdp"
# python -m evaluate.global.metrics --city_name="all" --model_name="gpt-4.1-mini" --prompt_type="normalized" --task_name="gdp"

# house price
# python -m evaluate.house_price.house_price_china --city_name="Shanghai" --mode="gen"
# python -m evaluate.house_price.house_price_us --city_name="all" --mode="eval" --model_name="google/gemma-3-12b-it" --prompt_type="normalized" --num_process=10
# python -m evaluate.house_price.metrics --city_name="all" --model_name="google/gemma-3-4b-it" --prompt_type="simple" 

# population
# python -m evaluate.global.global_indicator --city_name="all" --mode="eval" --model_name="gpt-4.1-mini" --prompt_type="simple" --num_process=10 --task_name="pop"
# python -m evaluate.global.metrics --city_name="all" --model_name="gpt-4.1-mini" --prompt_type="simple" --task_name="pop"

# # crime
# python -m evaluate.crime.crime_us --city_name="Chicago" --mode="gen"
# python -m evaluate.crime.crime_us --city_name="US" --mode="eval" --model_name="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8" --prompt_type="normalized" --task_name="violent"
# python -m evaluate.crime.metrics --city_name="US" --model_name="meta-llama/Llama-4-Maverick-17B-128E-Instruct-FP8" --prompt_type="normalized" --task_name="violent"

# drive
# python -m evaluate.transport.transport_us --city_name="SanFrancisco" --mode="gen"
# python -m evaluate.transport.transport_us --city_name="US" --mode="eval" --model_name="google/gemma-3-27b-it" --prompt_type="simple" --task_name="drive"
# python -m evaluate.transport.metrics --city_name="US" --model_name="google/gemma-3-27b-it" --prompt_type="simple"  --task_name="drive"

# public
# python -m evaluate.transport.transport_us --city_name="US" --mode="eval" --model_name="meta-llama/Llama-4-Scout-17B-16E-Instruct" --prompt_type="simple" --task_name="public"
# python -m evaluate.transport.metrics --city_name="US" --model_name="meta-llama/Llama-4-Scout-17B-16E-Instruct" --prompt_type="simple"  --task_name="public"

# mental health
# python -m evaluate.health.health_us --city_name="NewYork" --mode="gen"
# python -m evaluate.health.health_us --city_name="US" --mode="eval" --model_name="gpt-4.1-mini" --prompt_type="simple" --task_name="mental"
# python -m evaluate.health.metrics --city_name="US" --model_name="gpt-4.1-mini" --prompt_type="simple" --task_name="mental"

# life expectancy
# python -m evaluate.life_exp.life_exp_uk --city_name="Liverpool" --mode="gen"
# python -m evaluate.life_exp.life_exp_uk --city_name="UK" --mode="eval" --model_name="google/gemini-2.0-flash-001" --prompt_type="simple"
# python -m evaluate.life_exp.metrics --city_name="UK" --model_name="google/gemini-2.0-flash-001" --prompt_type="simple"

# # accessibility to health
# python -m evaluate.global.global_indicator --city_name="all" --mode="eval" --model_name="meta-llama/Llama-4-Scout-17B-16E-Instruct" --prompt_type="simple" --num_process=10 --task_name="acc2health"
# python -m evaluate.global.metrics --city_name="all" --model_name="meta-llama/Llama-4-Scout-17B-16E-Instruct" --prompt_type="simple" --task_name="acc2health"

# carbon
# python -m evaluate.global.global_indicator --city_name="all" --mode="eval" --model_name="google/gemma-3-4b-it" --prompt_type="simple" --num_process=10 --task_name="carbon"
# python -m evaluate.global.metrics --city_name="all" --model_name="google/gemma-3-4b-it" --prompt_type="simple" --task_name="carbon"

# building height
# python -m evaluate.global.global_indicator --city_name="all" --mode="eval" --model_name="gpt-4o" --prompt_type="simple" --num_process=10 --task_name="build_height"
# python -m evaluate.global.metrics --city_name="all" --model_name="gpt-4o" --prompt_type="simple" --task_name="build_height"

# feature
python -m evaluate.feature.extract_feature