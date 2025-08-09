import re

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory


def tokenize(text: str) -> list[str]:
    return [x.strip().lower() for x in re.findall(r"\w+(?:-\w+)*", text)]


def word_count(text: str) -> int:
    return len(tokenize(text))


class Stemmer:
    def __init__(self, vocab_sunda_path: str):
        # load vocabulary
        with open(vocab_sunda_path) as word_file:
            self.kamus = set(word.strip().lower() for word in word_file)

        # build dict
        self.awalan = [
            "ba",
            "barang",
            "di",
            "ka",
            "pa",
            "pada",
            "pang",
            "para",
            "per",
            "pi",
            "pika",
            "kapi",
            "sa",
            "sang",
            "si",
            "silih",
            "sili",
            "ti",
            "ting",
            "pating",
            "mi",
        ]
        self.akhiran = ["an", "eun", "keun", "na", "ing", "ning"]
        self.kata_ganti = ["p", "b"]
        self.kata_ganti2 = ["b", "d", "g", "h", "j", "l", "m", "n", "w", "y"]
        self.kata_ganti3 = ["c", "s"]
        self.konsonan = ["tr", "br", "cr", "kr", "pr", "jr", "dr"]

        # create sastrawi stemmer
        self.sastrawi = StemmerFactory().create_stemmer()

    def stem_ams(self, word: str):
        word = self.stem_purwoko(word)

        # sandhi vokal
        if re.search("ia", word):
            # print("ia")
            word = word.replace("ia", "é")
        if re.search("ae", word):
            # print("ae")
            word = word.replace("ae", "e")
        if re.search("aé", word):
            # print("aé")
            word = word.replace("aé", "é")
        if re.search("aa", word):
            # print("aa")
            word = word.replace("aa", "a")
        if re.search("au", word):
            # print("au ao")
            word = word.replace("au", "o")
        if re.search("ao", word):
            # print("au ao")
            word = word.replace("ao", "o")
            return word
        if re.search("ua", word):
            # print("ua")
            word = word.replace("ua", "o")
        if self.__is_in_dictionary(word):
            return word

        # sandhi konsonan
        if word.startswith("sing"):
            word = word.replace("sing", "")
        if word.startswith("si"):
            word = word.replace("si", "")
        if word.startswith("dang"):
            word = word.replace("dang", "")
        if word.startswith("da"):
            word = word.replace("da", "")
        if word.startswith("seung"):
            word = word.replace("seung", "")
        if word.startswith("seu"):
            word = word.replace("seu", "")

        if self.__is_in_dictionary(word):
            return word

        return word

    def stem_purwoko(self, word: str):
        old = word

        if self.__is_in_dictionary(word):
            return word

        # Hilangkan Awalan
        for prefix in self.awalan:
            if word.startswith(prefix):
                stemmed_word = word[len(prefix) :]
                if self.__is_in_dictionary(stemmed_word):
                    word = stemmed_word
                    break

        # Modul Nasal
        if word.startswith("m"):
            for pref in self.kata_ganti:
                morfoWord = word.replace("m", pref)
                if self.__is_in_dictionary(morfoWord):
                    word = morfoWord
                    break

        if word.startswith("n"):
            morfoWord2 = word.replace("n", "t", 1)
            if self.__is_in_dictionary(morfoWord2):
                word = morfoWord2

        if word.startswith("ng"):
            morfoWord3 = word.replace("ng", "k", 1)
            if self.__is_in_dictionary(morfoWord3):
                word = morfoWord3

        if word.startswith("nga"):
            words = word[len("nga") :]
            for pref in self.kata_ganti2:
                if words.startswith(pref):
                    if self.__is_in_dictionary(words):
                        word = words
                        break

        if word.startswith("ny"):
            for pref in self.kata_ganti3:
                morfoWord4 = word.replace("ny", pref)
                if self.__is_in_dictionary(morfoWord4):
                    word = morfoWord4
                    break

        # Hilangkan Akhiran
        for suffix in self.akhiran:
            if word.endswith(suffix):
                stemmed_word = word[: -len(suffix)]
                if self.__is_in_dictionary(stemmed_word):
                    word = stemmed_word
                    break

        # Hilangkan Sisipan
        # case huruf awalan l
        if word.startswith("l"):
            stemmed = word[len("l") :]
            if stemmed.startswith("al"):
                stemmed_word = stemmed.replace("al", "l")
                if self.__is_in_dictionary(stemmed_word):
                    word = stemmed_word

        # case huruf akhiran r
        if word.endswith("r"):
            stemmed_word = word.replace("al", "")
            if self.__is_in_dictionary(stemmed_word):
                word = stemmed_word

        # konsonan kata
        for infix in self.konsonan:
            if word.find(infix) != -1:
                stemmed_word = word.replace("al", "")
                if self.__is_in_dictionary(stemmed_word):
                    word = stemmed_word
                    break

        # case terdapat ar
        if "ar" in word:
            stemmed_word = word.replace("ar", "")
            if self.__is_in_dictionary(stemmed_word):
                word = stemmed_word

        # case terdapat in
        if "in" in word:
            stemmed_word = word.replace("in", "")
            if self.__is_in_dictionary(stemmed_word):
                word = stemmed_word

        # case terdapat um
        if "um" in word:
            stemmed_word = word.replace("um", "")
            if self.__is_in_dictionary(stemmed_word):
                word = stemmed_word

        # Hilangkan Barung
        for sufix in self.akhiran:
            if word.endswith(sufix):
                stemmed = word[: -len(sufix)]
                for prefix in self.awalan:
                    if stemmed.startswith(prefix):
                        stemmed_word = stemmed[len(prefix) :]
                        if self.__is_in_dictionary(stemmed_word):
                            word = stemmed_word
                            break

        # Hilangkan Bareng
        for prefix in self.awalan:
            if word.startswith(prefix) and word.endswith("na") and "dipika" in word:
                stemmed_word = (
                    word.replace(prefix, "").replace("na", "").replace("dipika", "")
                )
                if self.__is_in_dictionary(stemmed_word):
                    word = stemmed_word
                    break

            if self.__is_in_dictionary(word):
                return word

        return old

    def stem_sastrawi(self, word: str):
        return self.sastrawi.stem(word)

    def stem_ug18(self, word: str):
        # 1. Cek Kata di Kamus jika Ada SELESAI
        if self.__is_in_dictionary(word):  # Cek Kamus
            return word  # Jika Ada kembalikan

        # 2. Buang Infection suffixes (\-lah", \-kah", \-ku", \-mu", atau \-nya")
        word = self.__ug18_remove_inflection_suffixes(word)

        # 3. Buang Derivation suffix (\-i" or \-an")
        word = self.__ug18_remove_derivation_suffixes(word)

        # 4. Buang Derivation prefix
        word = self.__ug18_remove_derivation_prefix(word)

        return word

    # ----- INTERNAL METHODS

    def __is_in_dictionary(self, word):
        return word.lower() in self.kamus

    def __ug18_remove_inflection_suffixes(self, word):
        original_word = word
        if not re.search("[km]u|nya|[kl]ah|pun", word):  # Cek Inflection Suffixes
            new_word = re.sub("[km]u|nya|[kl]ah|pun", "", word)
            if not re.search(
                "[klt]ah|pun", word
            ):  # Jika berupa particles (“-lah”, “-kah”, “-tah” atau “-pun”)
                if not re.search(
                    "[km]u|nya", new_word
                ):  # Hapus Possesive Pronouns (“-ku”, “-mu”, atau “-nya”)
                    __kata__ = re.sub("[km]u|nya", "", new_word)
                    return __kata__
            return new_word
        return original_word

    def __ug18_remove_derivation_suffixes(self, word):
        original_word = word
        if re.search("(an|i)$", word):  # cek -kan
            new_word = re.sub("(an|i)$", "", word)
            if self.__is_in_dictionary(new_word):  # Cek Kamus
                return new_word
        if re.search("(kan)$", word):  # Cek Suffixes (mencari -kan dalam $kata)
            new_word = re.sub("(kan)$", "", word)
            if self.__is_in_dictionary(new_word):  # Cek Kamus
                return new_word
        return original_word

    def __ug18_remove_derivation_prefix(self, word):
        original_word = word

        # Tentukan Tipe Awalan
        if re.search(
            r"^(di|[ks]e)\S{1,}", word
        ):  # Jika di-,ke-,se- X{m,}, artinya elemen X harus ada minimal sebanyak m kali. cek apakah ada awalan di ke se (1 =ambil salah satu)
            new_word = re.sub(r"^(di|[ks]e)", "", word)

            if self.__is_in_dictionary(new_word):
                return new_word  # Jika ada balik

            __kata__ = self.__ug18_remove_derivation_suffixes(new_word)
            if self.__is_in_dictionary(__kata__):
                return __kata__

        if re.search(
            r"^([^aiueo])e\1[aiueo]\S{1,}", word, re.IGNORECASE
        ):  # aturan 37, “\S” untuk non-whitespace character /i agara dapat dteksi huruf kapital
            new_word = re.sub(r"^([^aiueo])e", "", word)

            if self.__is_in_dictionary(new_word):
                return new_word  # Jika ada balik

            __kata__ = self.__ug18_remove_derivation_suffixes(new_word)
            if self.__is_in_dictionary(__kata__):
                return __kata__

        return original_word
