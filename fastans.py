import os
import time
import re

class EncodingError(AttributeError):
    pass

class fastans:
    def __init__(self, stop_token):

        self.stop_token = stop_token
        
        self.i2t = {}
        self.t2i = {}
        self.mergers = {}

    def __str__(self):
        return (f"\nText Embedding layer:\nNumber of tokens: {len(self.t2i)}\n")

    # Main functions

    def train(self, text, num_mergers=10, printf=False, delay=1):
        # import counter
        from collections import Counter

        # join text, if his type is 'list'
        if isinstance(text, list):
            text = self.stop_token.join(text)
        else:
            if not self.stop_token.strip() in text:
                text += self.stop_token[:-1]

        # initialization clear i2t, t2i, mergers
        if self.t2i == {} or self.t2i == {} or self.mergers == {}:
            self.t2i = {str(i): i for i in range(256)}
            self.i2t = {i: [i] for i in range(256)}
            self.mergers = {}

        # prepare words:
        raw_words = text.lower().split(" ") # <- H -> h and spliting text by whitespace
        word_count = Counter(raw_words) # <- calculate count every word in text ^
        
        bpe_vocab = {}  # <- initialization bpe vocab

        # write word counts to the bpe vocab ^
        for word, count in word_count.items(): # <- data transformation: {word: count} -> [word, count]
            
            if word == "": # <- checking safety
                continue
            
            prep_word = " ".join(str(b) for b in word.encode("utf-8"))  # <- "word word" -> "byteword bytewrod"
            bpe_vocab[prep_word] = count # <- write ^ to the bpe vocab: "byteword": her count

        # write new data to the i2t, t2i and mergers
        if printf:
            start = time.perf_counter()
        for i in range(num_mergers): # <- cycle streeply for num mergers, which were be give
            if printf and i % delay == 0:
                print(f"\nFastrans Logs:\nMerge {i} in {time.perf_counter() - start} seconds\n")
                start = time.perf_counter()

            # get pairs: (byte, byte): their's count
            pairs = self.get_stats(bpe_vocab)
            if not pairs: # <- checking safety
                break

            best_pair = max(pairs, key=pairs.get) # get best pair, looking at count
            bpe_vocab = self.merge_vocab(best_pair, bpe_vocab)  # update bpe vocab: replace most often word(join): "bytewordbyteword"

            new_token = self._merge_name(best_pair)  # <- unique name for the pair, e.g. "104+101"
            self.mergers[best_pair] = new_token  # <- write pair
            
            new_id = max(self.t2i.values()) + 1  # <- calculate max token id in the t2i dictonary

            self.i2t[new_id] = self.i2t[self.t2i[best_pair[0]]] + self.i2t[self.t2i[best_pair[1]]] # <- write pair as list
            self.t2i[new_token] = new_id  # <- write pair

    def merge_name(self, pair):
        return pair[0] + "+" + pair[1]

    def add(self, token):
        # safety check
        if self.mergers == {}:
            raise EncodingError(
                "No rules were build for BPE")

        symbols = self.tetoby(token).split()
        if not symbols:
            raise EncodingError("Cannot add an empty token")

        for pair, merged in self.mergers.items():
            first, second = pair
            new_symbols = []
            i = 0
            n = len(symbols)
            while i < n:
                if i < n - 1 and symbols[i] == first and symbols[i+1] == second:
                    new_symbols.append(merged)
                    i += 2

                else:
                    new_symbols.append(symbols[i])
                    i += 1
            symbols = new_symbols

        while len(symbols) > 1:
            first, second = symbols[0], symbols[1]
            pair = (first, second)

            if pair in self.mergers:
                merged = self.mergers[pair]
            else:
                merged = self.merge_name(pair)
                self.mergers[pair] = merged
                new_id = max(self.t2i.values()) + 1
                self.i2t[new_id] = self.i2t[self.t2i[first]] + self.i2t[self.t2i[second]]
                self.t2i[merged] = new_id

            symbols = [merged] + symbols[2:]
            
        
        
    def encode(self, text):

        # safety check
        if self.mergers == {}:
            raise EncodingError(
                "No rules were build for BPE")

        # join text
        if isinstance(text, list):
            text = self.stop_token.join(text)

        prep_text = self.tetoby(text)
        tokens = prep_text.split()

        # apply every learned merge rule, in the order it was learned,
        # by scanning the token LIST and merging matching adjacent pairs
        # (instead of doing a raw substring replace on the joined string)
        for pair, merged in self.mergers.items():
            first, second = pair
            new_tokens = []
            i = 0
            n = len(tokens)
            while i < n:
                if i < n - 1 and tokens[i] == first and tokens[i + 1] == second:
                    new_tokens.append(merged)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens

        final_ids = [self.t2i.get(token, 0) for token in tokens]
        return final_ids
        
    def decode(self, ids):

        # safety check
        if self.mergers == {}:
            raise EncodingError(
                "No rules were build for BPE")

        raw_bytes = []
        for idx in ids:
            val = self.i2t.get(int(idx), [])
            raw_bytes.extend(val)

        return bytes(raw_bytes).decode("utf-8", errors="ignore")
        
    # Auxiliary functions

    def tetoby(self, text):
        return " ".join(str(b) for b in text.lower().encode("utf-8"))

    def _merge_name(self, pair):
        # Unique name for a merged pair. Uses a delimiter ("+") that can
        # never appear in a plain byte-id string ("0".."255"), so a merged
        # token name can never collide with a raw byte token name.
        return pair[0] + "+" + pair[1]

    def save(self, name):

        vocab = name
        with open(vocab + ".txt", "w", encoding="utf-8") as f:
            
            f.write("[mergers]\n")
            for k, v in self.mergers.items():
                key = " ".join(k) if isinstance(k, tuple) else str(k)
                f.write(f"{key} : {v}\n")

            f.write("[t2i]\n")
            for k, v in self.t2i.items():
                f.write(f"{k} : {v}\n")

            f.write("[i2t]\n")
            for k, v in self.i2t.items():
                val = " ".join(map(str, v))
                f.write(f"{k} : {val}\n")

    @classmethod
    def load(cls, obj, name):
        
        if not os.path.exists(name + '.txt'):
            raise FileNotFoundError(f"File with the BPE layer '{name}.txt' was not found")
        
        vocab = name
        mergers, t2i, i2t = {}, {}, {}
        with open(vocab + ".txt", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line: continue

                if line in ["[mergers]", "[t2i]", "[i2t]"]:
                    sel = line
                    continue
                k, v = line.split(" : ")
                if sel == "[mergers]":
                    mergers[tuple(k.split())] = v
                elif sel == "[t2i]":
                    t2i[k] = int(v)
                elif sel == "[i2t]":
                    i2t[int(k)] = [int(x) for x in v.split()]
                    
        obj.mergers = mergers
        obj.t2i = t2i
        obj.i2t = i2t          
        return obj

    def get_stats(self, vocab):
        pairs = {}

        for word, freq in vocab.items():
            symbols = word.split()

            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i+1])

                pairs[pair] = pairs.get(pair, 0) + freq
        return pairs

    def merge_vocab(self, pair, vocab):
        new_vocab = {}
        first, second = pair

        for word, freq in vocab.items():
            symbols = word.split()
            merged_symbols = []
            i = 0
            n = len(symbols)
            # scan the token LIST and merge only exact adjacent matches,
            # instead of a substring replace on the joined string
            while i < n:
                if i < n - 1 and symbols[i] == first and symbols[i + 1] == second:
                    merged_symbols.append(self._merge_name(pair))
                    i += 2
                else:
                    merged_symbols.append(symbols[i])
                    i += 1
            new_word = " ".join(merged_symbols)
            new_vocab[new_word] = new_vocab.get(new_word, 0) + freq
        return new_vocab
    
