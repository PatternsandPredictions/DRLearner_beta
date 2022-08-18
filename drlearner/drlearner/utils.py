import jax.nn
import jax.numpy as jnp


def epsilon_greedy_prob(q_values, epsilon):
    """Get probability of actions under epsilon-greedy policy provided estimated q_values"""
    num_actions = q_values.shape[0]
    max_action = jnp.argmax(q_values)
    probs = jnp.full_like(q_values, fill_value=epsilon / num_actions)
    probs = probs.at[max_action].set(1 - epsilon * (num_actions - 1) / num_actions)
    return probs


def get_beta(mixture_idx: jnp.ndarray, beta_min: float, beta_max: float, num_mixtures: int):
    beta = jnp.linspace(beta_min, beta_max, num_mixtures)[mixture_idx]
    return beta


def get_gamma(mixture_idx: jnp.ndarray, gamma_min: float, gamma_max: float, num_mixtures: int):
    gamma = jnp.linspace(gamma_min, gamma_max, num_mixtures)[mixture_idx]
    return gamma


def get_epsilon(actor_id: int, epsilon_base: float, num_actors: int, alpha: float = 8.0):
    """Get epsilon parameter for given actor"""
    epsilon = epsilon_base ** (1 + alpha * actor_id / ((num_actors - 1) + 0.0001))
    return epsilon


def get_beta_ngu(mixture_idx: jnp.ndarray, beta_min: float, beta_max: float, num_mixtures: int):
    """Get beta parameter for given number of mixtures and mixture_idx"""
    beta = jnp.where(
        mixture_idx == num_mixtures - 1,
        beta_max,
        beta_min + beta_max * jax.nn.sigmoid(10 * (2 * mixture_idx - (num_mixtures - 2)) / (num_mixtures - 2))
    )
    return beta


def get_gamma_ngu(mixture_idx: jnp.ndarray, gamma_min: float, gamma_max: float, num_mixtures: int):
    """Get gamma parameters for given number of mixtures in descending order"""
    gamma = 1 - jnp.exp(
        ((num_mixtures - 1 - mixture_idx) * jnp.log(1 - gamma_max) +
         mixture_idx * jnp.log(1 - gamma_min)) / (num_mixtures - 1))
    return gamma

