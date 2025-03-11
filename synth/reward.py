from synth.simulations import generate_simulations
from synth.price_data_provider import PriceDataProvider
from synth.simulation_input import SimulationInput
from synth.crps_calculation import calculate_crps_for_miner
from synth.utils.helpers import get_intersecting_arrays

from typing import List
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Example log messages
# logging.info("This is an info message")
# logging.warning("This is a warning message")
# logging.error("This is an error message")
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
import numpy as np


def compute_softmax(score_values: np.ndarray, beta: float) -> np.ndarray:
    # Mask out invalid scores (e.g., -1)
    mask = score_values != -1  # True for values to include in computation

    logging.info(f"Going to use the following value of beta: {beta}")

    # Compute softmax scores only for valid values
    exp_scores = np.exp(beta * score_values[mask])
    softmax_scores_valid = exp_scores / np.sum(exp_scores)

    # Create final softmax_scores with 0 where scores were -1
    softmax_scores = np.zeros_like(score_values, dtype=float)
    softmax_scores[mask] = softmax_scores_valid

    return softmax_scores


def clean_numpy_in_crps_data(crps_data: []) -> []:
    cleaned_crps_data = [
        {
            key: (float(value) if isinstance(value, np.float64) else value)
            for key, value in item.items()
        }
        for item in crps_data
    ]
    return cleaned_crps_data


def reward(    
    price_data_provider: PriceDataProvider,
    simulation_input: SimulationInput    
):
    predictions = generate_simulations(
        asset=simulation_input.asset,        
        start_time=simulation_input.start_time,
        time_increment=simulation_input.time_increment,
        time_length=simulation_input.time_length,
        num_simulations=simulation_input.num_simulations,
        sigma=0.001,        
    )
    end_time = predictions[0][len(predictions[0]) - 1]["time"]
    real_prices = price_data_provider.fetch_data(end_time)
    
    if len(real_prices) == 0:
        return -1, [], []
    
    # in case some of the time points is not overlapped
    intersecting_predictions = []
    intersecting_real_price = real_prices
    for prediction in predictions:
        intersecting_prediction, intersecting_real_price = (
            get_intersecting_arrays(prediction, intersecting_real_price)
        )
        intersecting_predictions.append(intersecting_prediction)

    predictions_path = [
        [entry["price"] for entry in sublist]
        for sublist in intersecting_predictions
    ]
    real_price_path = [entry["price"] for entry in intersecting_real_price]

    try:
        score, detailed_crps_data = calculate_crps_for_miner(
            np.array(predictions_path).astype(float),
            np.array(real_price_path),
            simulation_input.time_increment,
        )
    except Exception as e:
        logging.error(
            f"Error calculating CRPS for miner : {e}"
        )
        return -1, [], []

    return score, detailed_crps_data, real_prices

def get_rewards(
    price_data_provider:PriceDataProvider,
    simulation_input:SimulationInput,
    uids:List[int],
    softmax_beta:float
)->(np.ndarray, []):
    """
    Returns an array of rewards for the given query and responses.

    Args:
    - query (int): The query sent to the miner.
    - responses (List[float]): A list of responses from the miner.

    Returns:
    - np.ndarray: An array of rewards for the given query and responses.
    """
    logging.info(f"In rewards, uids={uids}")
    scores = []
    detailed_crps_data_list = []
    real_prices_list = []
    prediction_id_list = []
    for i, id in enumerate(uids):
        score, detailed_crps_data, real_prices = reward(
            price_data_provider,
            simulation_input
        )
        scores.append(score)
        detailed_crps_data_list.append(detailed_crps_data)
        real_prices_list.append(real_prices)
        prediction_id_list.append(id)
    
    score_values = np.array(scores)
    softmax_scores = compute_softmax(score_values, softmax_beta)
    
    return scores, softmax_scores
    
    
