import csv
import re
import sys
from six.moves.html_parser import HTMLParser

from bs4 import BeautifulSoup

from emoint.utils.utils import list_files


def clean_text(text):
    return re.sub("\n|\t", " ", text).encode('utf-8')


def depth(comment, data):
    if comment['parent-url'] == "-1":
        return 1
    else:
        return 1 + depth(data.find(url=comment['parent-url']), data)


def get_pairs(data):
    tuples = []
    title = clean_text(data.title.text)
    for comment in data.find_all('comment'):
        comment_text = clean_text(comment.text)
        if comment['parent-url'] == '-1':
            tuples.append(['none', comment_text, title, depth(comment, data)])
        else:
            parent_comment = data.find(url=comment['parent-url'])
            parent_comment_text = clean_text(parent_comment.text)
            # If R, Q paris are from same author => continuation of first
            if clean_text(parent_comment.user.text) == clean_text(comment.user.text):
                # Todo - does this pair belong to None category or Agree or just discard these
                tuples.append(['none', comment_text, parent_comment_text, depth(comment, data)])
            else:
                # If R, Q pairs are from different authors and opposite side => Disagree
                if comment['side'] != parent_comment['side']:
                    tuples.append(['disagree', comment_text, parent_comment_text, depth(comment, data)])
                else:
                    # If R, Q pairs are from different authors and opposite side => Disagree
                    tuples.append(['agree', comment_text, parent_comment_text, depth(comment, data)])
    return tuples


def prepare(path, filename):
    h = HTMLParser()
    with open(filename, 'w') as csvfile:
        writer = csv.writer(csvfile, quoting=csv.QUOTE_ALL, delimiter='\t')
        for f in list_files(path, lambda x: x.endswith('xml')):
            data = BeautifulSoup(h.unescape(open(f).read()))
            pairs = get_pairs(data)
            writer.writerows(pairs)


if __name__ == '__main__':
    if sys.argv[1] == '1':
        prepare('/home/venkatesh/create_debate/training/',
            '/home/venkatesh/create_debate/train_64.txt')
    if sys.argv[1] == '2':
        prepare('/home/venkatesh/create_debate/development/',
            '/home/venkatesh/create_debate/dev_64.txt')
    else:
        prepare('/home/venkatesh/create_debate/testing/',
            '/home/venkatesh/create_debate/test_64.txt')

