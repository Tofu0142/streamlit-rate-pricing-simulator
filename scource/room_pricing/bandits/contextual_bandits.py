import json
import time
import numpy as np
import vowpalwabbit
from room_pricing.bandits.utils import get_expected_revenue


class ContextualMAB:
    def __init__(
        self,
        df,
        context_columns,
        cost_columns,
        num_actions,
        min_action,
        max_action,
        epsilon,
        **kwargs,
    ):
        self.df = df.copy()

        self.context_columns = context_columns

        self.X = self.df[context_columns].to_dict(orient="records")
        self.Y = self.df[cost_columns].values

        self.ACTIONS = np.linspace(min_action, max_action, num_actions)

        self.vw = vowpalwabbit.Workspace(
            f"--cb_explore {str(num_actions)}"
            + f" --epsilon {epsilon}"
            + f" --quiet"
            + f" --coin --chain_hash"
            + f" -q ::"
        )

    def to_vw_example_format(self, context, cats_label=None):
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

    def get_proposed_price(
        self,
        action,
        Y,
    ):
        nightly_rate, observed_price, converted = Y

        price = nightly_rate * (1 + action)

        return price

    def get_cost_and_price(
        self,
        action,
        Y,
        type,
        alpha=10,
        beta=1,
        upper_elasticity=None,
        lower_elasticity=None,
    ):
        nightly_rate, observed_price, converted = Y

        price = nightly_rate * (1 + action)

        cost = -get_expected_revenue(
            price,
            observed_price,
            converted,
            type,
            alpha,
            beta,
            upper_elasticity,
            lower_elasticity,
        )

        return cost, price

    def predict_action(self, context):
        vw_text_example = self.to_vw_example_format(context)
        pmf = np.array(self.vw.predict(vw_text_example))
        pmf /= pmf.sum()  # Normalise pmf
        # print(pmf)
        # time.sleep(0.1)
        idx = np.random.choice(len(self.ACTIONS), p=pmf)
        return idx, pmf[idx]

    def run(
        self,
        type="elasticity",
        alpha=None,
        beta=None,
        upper_elasticity=None,
        lower_elasticity=None,
        learn=True,
        random=False,
    ):
        history = {}

        rewards = []
        actions = []
        prices = []

        for i in range(len(self.X)):
            context = self.X[i]
            #  if i % 1000 == 0:
            #      print(self.predict_action(self.vw, self.X[0]))

            if not random:
                idx, pmf = self.predict_action(context)
                action = self.ACTIONS[idx]
            else:
                idx = np.random.choice(len(self.ACTIONS))
                action = self.ACTIONS[idx]
                pmf = 1 / len(self.ACTIONS)

            actions.append(action)

            # 4. Get cost of the action we chose
            cost, price = self.get_cost_and_price(
                action, self.Y[i], type, alpha, beta, upper_elasticity, lower_elasticity
            )
            rewards.append(-cost)
            prices.append(price)

            if learn:
                # if i % 1000 == 0:
                #     print("Learning")
                example = self.to_vw_example_format(
                    context,
                    cats_label=(
                        idx + 1,
                        cost,
                        pmf,
                    ),  # idx needs to be increased by 1 according to VW counting
                )

                # print(example)

                # vw_format = self.vw.parse(txt_ex)

                self.vw.learn(example)

        print("*" * 100)

        history["rewards"] = rewards
        history["actions"] = actions
        history["prices"] = prices
        history["agent"] = self.vw

        return history
