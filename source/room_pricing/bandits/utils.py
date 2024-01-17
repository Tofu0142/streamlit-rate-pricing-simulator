import numpy as np
import matplotlib.pyplot as plt


def to_vw_example_format(context, cats_label=None):
    output = "| "

    for k, v in context.items():
        if k == "supplier":
            output += f"{k}={str(v)}:1 "
        else:
            output += f"{k}:{v} "

    if cats_label is not None:
        chosen_action, cost, pdf_value = cats_label
        output = f"{chosen_action}:{cost}:{pdf_value} {output}"

    return output[:-1]  # remove last blank space


def get_expected_revenue(
    price,
    observed_price,
    converted,
    type="polynomial",
    alpha=None,
    beta=None,
    upper_elasticity=None,
    lower_elasticity=None,
):
    if type == "polynomial":
        if converted:
            max_price = observed_price * (1 + alpha)
        else:
            max_price = observed_price * (1 - beta)

        return price / (1 + np.exp(price - max_price))

    elif type == "elasticity":
        if converted:
            if price <= observed_price * (1 + upper_elasticity):
                return price
            else:
                return 0
        else:
            if price <= observed_price * (1 - lower_elasticity):
                return price
            else:
                return 0
    else:
        raise ValueError("Only 'polynomial' or 'elasticity' are valid types")


def plot_expected_revenue(observed_price, converted, alpha, beta):
    p0 = observed_price
    i = np.linspace(0, p0 * 3, 100)
    plt.plot(
        i,
        [
            get_expected_revenue(
                price=_i, observed_price=p0, converted=converted, alpha=alpha, beta=beta
            )
            for _i in i
        ],
    )


def plot_reward_rate(rewards, label=None):
    num_iterations = len(rewards)
    idx = range(1, num_iterations + 1)
    reward_rate = np.cumsum(rewards) / idx

    plt.plot(idx, reward_rate, label=label)
    plt.xlabel("num_iterations", fontsize=14)
    plt.ylabel("reward rate", fontsize=14)
    if label is not None:
        plt.legend()
