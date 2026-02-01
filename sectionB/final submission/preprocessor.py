"""
File to handle data preprocessing tasks for our twi corpus
"""

import re

class Text:
    def __init__(self, path: str):
        self.sentences = self.__load_text(path)
        self.sentence_tokens, self.word_tokens = self.tokenize()
        
    
    def __load_text(self, path):
        try:
            with open(path, 'r', encoding='utf-8') as file:
                temp = str()

                for line in file.readlines():
                    temp += '<s> ' + line.strip() + ' </s>\n'

            return temp.split('\n')

        except FileNotFoundError:
            print(f"File not found: {path}")
            return ""


    def tokenize(self):
        sentence_tok = []
        word_tok = []
        for _, text in enumerate(self.sentences):
            text = text.lower()
            text = re.sub(pattern=r'[^a-zɛɔ</>\s]', repl='', string=text)
            temp = text.split()
            
            sentence_tok.append(temp)
            word_tok.extend(temp)
        return sentence_tok[:-1], word_tok[:-1]



if __name__ == '__main__':
    import sys
    args = sys.argv[1:]   # skip script name
    
    test = Text(args[0])
