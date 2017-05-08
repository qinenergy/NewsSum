from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import argparse
import gnp


def decode_g_result(g):


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--query', type=str, default=None, 
                        help='news query')
    args = parser.parse_args()
    if args.query == None:
        print(Please input a query using --query "Something to search")
    g_result = gnp.get_google_news_query("What's happening on earth")
    decode_g_result(g_result)
