import logging
import os

from gcloud import storage


def list_files(base_path, predicate):
    if base_path.startswith("gs://"):
        prefix = 'gs://'
        bucket_name = base_path.split('/')[2]
        client = storage.Client()
        bucket = client.get_bucket(bucket_name)
        for f in bucket.list_blobs():
            if predicate(f.name):
                logging.info("Found file: {}".format(prefix + bucket_name + '/' + f.name))
                yield prefix + bucket_name + '/' + f.name
    else:
        for folder, subs, files in os.walk(base_path):
            for filename in files:
                if predicate(os.path.join(folder, filename)):
                    yield (os.path.join(folder, filename))


class LIWCTrie:
    def __init__(self):
        self.root = dict()
        self.end = '$'
        self.cont = '*'

    def insert(self, word, categories):
        """
        Insert liwc word and categories in trie
        :param word: word to insert
        :param categories: liwc categories
        :return: None
        """
        if len(word) == 0:
            return
        curr_dict = self.root
        for letter in word[:-1]:
            curr_dict = curr_dict.setdefault(letter, {})
        if word[-1] != self.cont:
            curr_dict = curr_dict.setdefault(word[-1], {})
            curr_dict[self.end] = categories
        else:
            curr_dict[self.cont] = categories

    def in_trie(self, word):
        """
        Query if a word is in trie or not
        :param word: query word
        :return: True if word is present otherwise False
        """
        curr_dict = self.root
        for letter in word:
            if letter in curr_dict:
                curr_dict = curr_dict[letter]
            elif self.cont in curr_dict:
                return True
            else:
                return False
        if self.cont in curr_dict or self.end in curr_dict:
            return True
        else:
            return False

    def get(self, word):
        """
        get value stored against word
        :param word: word to search
        :return: list of categories if word is in query otherwise None
        """
        curr_dict = self.root
        for letter in word:
            if letter in curr_dict:
                curr_dict = curr_dict[letter]
            elif self.cont in curr_dict:
                return curr_dict[self.cont]
            else:
                return None
        if self.cont in curr_dict or self.end in curr_dict:
            if self.cont in curr_dict:
                return curr_dict[self.cont]
            else:
                return curr_dict[self.end]
        else:
            return None

    def __getitem__(self, item):
        return self.get(item)

    def __contains__(self, item):
        return self.in_trie(item)



