from instance import Instance


# ++++++++++++
# | Settings |
# ++++++++++++

conf = {
    # data
    'mu': 100,  # sale mean
    'sigma': 100,  # sale standard deviation
    'days': 365,  # total number of sale days
    # parameters
    'dio': 15,  # days inventory outstanding
    'lt': 30,  # lead time
    # constraints
    'moq': 0,  # minimal order quantity
    'mpf': 7,  # minimal purchase frequency
    'sc': 5000,  # stock capacity
    'msl': 0.95,  # minimal service level
    'mxsl': 0.99  # maximal service level
}


def run():
    ins = Instance(conf)
    ins.evaluate_point_prediction()
    ins.run()
    ins.plot()


if __name__ == '__main__':
    run()

