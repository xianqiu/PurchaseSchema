from instance import Instance, generate_sale_data
from copy import deepcopy


# ++++++++++++
# | Settings |
# ++++++++++++

conf1 = {
    # data
    'mu': 100,  # sale mean
    'sigma': 100,  # sale standard deviation
    'days': 365,  # total number of sale days
    'price': 200,  # sale price
    'cost': 100,  # purchase cost
    'inv_cost': 4,  # inventory cost per day
    # objective
    'opp': 15,  # days inventory outstanding
    # constraints
    'lt': 30,  # lead time
    'moq': 0,  # minimal order quantity
    'mpf': 10,  # minimal purchase frequency
    'sc': 3000,  # stock capacity
    'msl': 0.05,  # minimal service level
    'mxsl': 0.99,  # maximal service level
    'sfl': 0.95,  # safety level (the probability of not overloaded)
}

conf2 = deepcopy(conf1)
conf2['msl'] = 0.99
conf2['sfl'] = 0


shared_sale = generate_sale_data(conf1['mu'], conf1['sigma'], conf1['days'])


def run1():
    """ Run with restricted service level.
    """
    ins = Instance(conf1)
    ins.sale = shared_sale
    ins.evaluate_point_prediction()
    ins.run()
    ins.plot()


def run2():
    """ Run with high service level.
    """
    ins = Instance(conf2)
    ins.sale = shared_sale
    ins.evaluate_point_prediction()
    ins.run()
    ins.plot()


if __name__ == '__main__':
    """ Question: What do we benefit from restricting service level?
        Answer: Money.
    """
    run1()
    run2()

