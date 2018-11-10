import time
import statistics

import numpy as np
import scipy.stats as ss
import matplotlib.pyplot as plt


def generate_sale_data(mu, sigma, size):
    """ Sales data generates from N(mu, sigma^2).
    :param mu: mean of the normal distribution
    :param sigma: standard error of normal distribution
    :param size: data size
    :return: list
    """
    np.random.seed(int(time.time()))
    data = np.random.normal(mu, sigma, size)
    return [max(round(x), 0) for x in data.tolist()]


class Instance(object):
    """ Instance for simulation.
    """

    def __init__(self, conf):
        self.conf = conf
        self.sale = generate_sale_data(self.conf['mu'], self.conf['sigma'], self.conf['days'])
        self.stock = [0] * self.conf['days']
        self.stock_cache = {}
        self.order_history = [0] * self.conf['days']
        self.order_count = 0
        self.T = self.conf['dio']  # sale period
        self.stockout_days = 0
        self.stockfull_days = 0
        self.alpha = self.__calculate_service_level()

    def __get_stock(self, day):
        return self.stock[day]

    def __get_sale(self, day):
        return self.sale[day]

    def __init_stock(self):
        self.stock[0] = self.conf['sc']

    def __order(self, day):
        q = self.__calculate_order_quantity(day)
        if q > 0:
            self.stock_cache[day+self.conf['lt']] = q
            self.order_history[day] = q
            self.order_count += 1

    def __in_stock(self, day):
        if day in self.stock_cache.keys():
            self.stock[day] += self.stock_cache[day]
            del self.stock_cache[day]
        if self.__get_stock(day) > self.conf['sc']:
            self.stockfull_days += 1

    def __out_stock(self, day):
        s = self.__get_stock(day) - self.__get_sale(day)
        if s < 0:
            s = 0
            if day >= self.conf['lt']:
                self.stockout_days += 1
        if day < self.conf['days'] - 1:
            self.stock[day+1] = s

    def __calculate_service_level(self):
        try:
            cu = self.conf['price'] - self.conf['cost'] - (self.T + 1)/2
            co = self.conf['inv_cost'] * self.T
            critical_ratio = cu/(cu+co)
            if critical_ratio < self.conf['msl']:
                return self.conf['msl']
            if critical_ratio > self.conf['mxsl']:
                return self.conf['mxsl']
            return critical_ratio
        except KeyError:
            return self.conf['msl']

    def __calculate_order_quantity(self, day):
        # satisfy minimal purchase frequency
        if day < self.conf['mpf'] or sum(self.order_history[day-self.conf['mpf']+1: day]) > 0:
            return 0
        # calculate demand of one period
        z = ss.norm.ppf(self.alpha)
        d = self.T * self.conf['mu'] + np.sqrt(self.T) * self.conf['sigma'] * z
        # do not exceed stock capacity with a probability equal to "safety level"
        if 'sfl' in self.conf.keys():
            d_max = self.__calculate_max_demand(day)
            if d > d_max:
                d = d_max
        # calculate the total demand in one period (including the incoming stock)
        s = self.__get_stock(day)
        for key in self.stock_cache:
            if day < key <= day + self.T:
                s += self.stock_cache[key]
        # calculate the stock on day+LT under the current service level alpha
        s1 = max(s - d, 0)
        # calculate the theoretical order quantity for the whole DIO.
        q = max(d - s1, 0)
        # satisfy MOQ
        if q < self.conf['moq']:
            return 0
        # spread the quantity over and make sure that q >= MOQ
        q = q / np.ceil(self.T / self.conf['mpf'])
        if q < self.conf['moq']:
            return self.conf['moq']

        return int(q)

    def __calculate_max_demand(self, day):
        # calculate the total demand in one period (including the incoming stock)
        s = self.__get_stock(day)
        for key in self.stock_cache:
            if day < key <= day + self.T:
                s += self.stock_cache[key]
        # calculate the max demand of one period
        z = ss.norm.ppf(1-self.conf['sfl'])
        d = self.T * self.conf['mu'] + self.conf['sc'] - s + np.sqrt(self.T) * self.conf['sigma'] * z
        return max(d, 0)

    def run(self):
        self.__init_stock()
        self.__out_stock(0)
        for day in range(1, self.conf['days']):
            self.__in_stock(day)
            self.__order(day)
            self.__out_stock(day)
        self.evaluate()

    def evaluate(self):
        # simple model
        print("==== Performance Evaluation ====")
        print('service level: %.2f' % self.alpha)
        print('stockout days (after the first LT): %d -- rate: %.2f' %
              (self.stockout_days, self.stockout_days/(self.conf['days']-self.conf['lt'])))
        print('stockfull days: %d -- rate: %.2f' % (self.stockfull_days, self.stockfull_days/self.conf['days']))
        stock_mean = statistics.mean(self.stock)
        stock_stdev = statistics.stdev(self.stock)
        print('average stock: %d -- standard deviation: %d' % (stock_mean, stock_stdev))
        sale_mean = statistics.mean(self.sale)
        print('average DIO: %.2f -- objective DIO: %d' % (stock_mean/sale_mean, self.conf['dio']))
        print('purchase frequency (PF): %.2f -- minimal PF: %d' % (self.conf['days']/self.order_count, self.conf['mpf']))
        # news vendor model
        if 'price' not in self.conf.keys():
            return
        self.__evaluate_profit()

    def __evaluate_profit(self):
        inv_cost = sum(self.stock) * self.conf['inv_cost']
        gmv = sum(self.sale) * self.conf['price']
        cost = sum(self.sale) * self.conf['cost']
        profit = gmv - cost - inv_cost
        print("==== Profit Analysis ====")
        print("GMV: %d" % gmv)
        print("total profit: %d" % profit)
        print("inventory cost: %d" % inv_cost)

    def plot(self):
        plt.figure(1, figsize=(8, 12))
        plt.subplots_adjust(hspace=0.4)
        ax = plt.subplot(311)
        ax.set_title('Sales')
        ax.plot(self.sale)

        ax = plt.subplot(312)
        ax.set_title('Orders')
        ax.bar(range(self.conf['days']), self.order_history)

        ax = plt.subplot(313)
        ax.set_title('Stocks')
        ax.plot(self.stock)
        ax.plot([self.conf['sc']] * self.conf['days'])

        plt.show()

    def evaluate_point_prediction(self):
        mean = statistics.mean(self.sale)
        print("==== Point Prediction ====")
        print('optimal point: x = %d' % mean)
        print('Error by MAPE-7: %.2f' % self.__mape_of_point_estimation(mean, 7))
        print('Error by MAPE-14: %.2f' % self.__mape_of_point_estimation(mean, 14))

    def __mape_of_point_estimation(self, point, agg=1):
        sale_agg = self.__aggregate(self.sale, agg)
        err = []
        for x in sale_agg:
            if x == 0:
                return np.Inf
            else:
                err.append(abs(x - point)/x)
        return np.mean(err)

    @staticmethod
    def __aggregate(array, agg):
        k = int(np.ceil(len(array)/agg))
        res = [sum(array[i*agg: (i+1)*agg]) for i in range(k)]
        return res

