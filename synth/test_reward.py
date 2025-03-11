# import schedule
import time
import json
from synth.simulation_input import SimulationInput
from synth.utils.helpers import from_iso_to_unix_time
from synth.reward import reward, get_rewards
from synth.price_data_provider import PriceDataProvider
from synth.price_simulation import simulate_single_price_path
from synth.simulations import generate_simulations

def test_get_reward():
    miner_id = 0
    start_time = "2024-11-26T00:00:00+00:00"
    scored_time = "2024-11-28T00:00:00+00:00"
    print(f"start_time={from_iso_to_unix_time(start_time)}")
    print(f"scored_time={from_iso_to_unix_time(scored_time)}")
    print("--------------------------")
    
    price_data_provider = PriceDataProvider("BTC")
    simulation_input = SimulationInput(
        asset="BTC",
        start_time=start_time,
        time_increment=300,
        time_length=86400,
        num_simulations=1,
        
    )
    # reward_value = reward(price_data_provider, simulation_input)
    reward_value, softmax_scores = get_rewards(price_data_provider, simulation_input, [0,1,2], -0.002)
    print("----------------------\n")
    print(f"reward_value={reward_value}")
    print(f"softmax_scores={softmax_scores}")

# schedule.every(10).seconds.do(my_function)   
def another_test():
    print("Gamj") 

data = generate_simulations(
    asset="BTC",
    start_time = "2024-11-26T00:00:00+00:00",
    time_increment=300,
    time_length = 86400,
    num_simulations = 1,
    sigma=0.01
)
with open('data.json', 'w') as json_file:
    json.dump(data, json_file, indent=4)
    
while True:
    test_get_reward()
    # another_test()
    time.sleep(10)    
    

    
    