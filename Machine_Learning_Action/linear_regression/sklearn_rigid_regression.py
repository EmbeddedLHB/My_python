#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from bs4 import BeautifulSoup
from sklearn import linear_model
from ridge_regression import ridge_test


def scrape_page(ret_x, ret_y, file, year, num_pieces, original_price):
    """Read data from pages, and generate ret_x and ret_y list

    Args:
        ret_x: dataset x
        ret_y: dataset y
        file: The HTML file
        year: The year
        num_pieces: The number of lego pieces
        original_price: The original price
    Returns:
        无
    """
    with open(file, encoding='utf-8') as f:
        html = f.read()
    soup = BeautifulSoup(html)
    i = 1
    # Parsing according to the structure of the HTML page
    current_row = soup.find_all('table', r="%d" % i)
    while len(current_row) != 0:
        current_row = soup.find_all('table', r="%d" % i)
        title = current_row[0].find_all('a')[1].text
        lwr_title = title.lower()
        # Find out if there is a new label
        if lwr_title.find('new') > -1 or lwr_title.find('nisb') > -1:
            new_flag = 1.0
        else:
            new_flag = 0.0

        # Only to collect data that has already been sold
        sold_unicde = current_row[0].find_all('td')[3].find_all('span')
        if len(sold_unicde) == 0:
            print("Commodity #%d not for sale" % i)
        else:
            # Parsing page to get the current price
            sold_price = current_row[0].find_all('td')[4]
            price_str = sold_price.text
            price_str = price_str.replace('$', '')
            price_str = price_str.replace(',', '')
            if len(sold_price) > 1:
                price_str = price_str.replace('Free shipping', '')
            selling_price = float(price_str)
            # Remove the incomplete package peice
            if selling_price > original_price * 0.5:
                print("%d\t%d\t%d\t%f\t%f" % (year, num_pieces, new_flag, original_price, selling_price))
                ret_x.append([year, num_pieces, new_flag, original_price])
                ret_y.append(selling_price)
        i += 1
        current_row = soup.find_all('table', r="%d" % i)


def set_data_collect(ret_x, ret_y):
    """Read the data of the six lego sets in turn and generate a data matrix
    x year's lego a, the number of parts b, the original price c

    """
    scrape_page(ret_x, ret_y, './lego/lego8288.html', 2006, 800, 49.99)
    scrape_page(ret_x, ret_y, './lego/lego10030.html', 2002, 3096, 269.99)
    scrape_page(ret_x, ret_y, './lego/lego10179.html', 2007, 5195, 499.99)
    scrape_page(ret_x, ret_y, './lego/lego10181.html', 2007, 3428, 199.99)
    scrape_page(ret_x, ret_y, './lego/lego10189.html', 2008, 5922, 299.99)
    scrape_page(ret_x, ret_y, './lego/lego10196.html', 2009, 3263, 249.99)


def sklearn_linear_regression():
    w = linear_model.Ridge(alpha=.5)
    lg_x = []
    lg_y = []
    set_data_collect(lg_x, lg_y)
    w.fit(lg_x, lg_y)
    print("%f + %f * year + %f * part's num + %f * whether is completely new + %f * original price"
          % (w.intercept_, w.coef_[0], w.coef_[1], w.coef_[2], w.coef_[3]))


if __name__ == '__main__':
    sklearn_linear_regression()
