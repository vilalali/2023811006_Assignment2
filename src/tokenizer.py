import regex


class Tokenizer:

    def replace_url(self, file_content):
        url = regex.compile(
            r"(?:https?:\/\/|www)[-a-zA-Z0-9:._]{1,256}\.[a-zA-Z0-9]{1,6}\b([-a-zA-Z0-9@:%_\+.~#?&//=]*)"
        )
        return url.sub("<URL>", file_content)

    def replace_email(self, file_content):
        email = regex.compile(r"\b[a-zA-Z0-9._-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,4}\b")
        return email.sub("<MAILID>", file_content)

    def replace_hashtags(self, file_content):
        hashtags = regex.compile(r"(?:(?<=\s)|^)#(\w+)")
        return hashtags.sub("<HASHTAG>", file_content)

    def replace_mensions(self, file_content):
        mension = regex.compile(r"(?<=^|[^\/])(@[A-Za-z0-9_.]{3,25})")
        return mension.sub("<MENSION>", file_content)

    def replace_date(self, file_content):
        ## Jan 24th 2025
        date1 = regex.compile(
            r"(\b\d{1,2}(\w{2})?\W{0,2}\b)\b(January|February|March|April|May|June|July|August|September|October|Novermber|December)\b(\W{0,2}\d{4}\b)"
        )
        ## 24th Jan 2025
        date2 = regex.compile(
            r"\b(January|February|March|April|May|June|July|August|September|October|Novermber|December)\b(\b\W{0,2}\d{1,2}(\w{2})?\W{0,2}\b)(\W{0,2}\d{4}\b)"
        )
        ## Jan 2025
        date3 = regex.compile(
            r"\b(January|February|March|April|May|June|July|August|September|October|Novermber|December)\b\s*(\W{0,2}\d{4}\b)"
        )
        ## Jan 24th
        date4 = regex.compile(
            r"\b(January|February|March|April|May|June|July|August|September|October|Novermber|December)\b\s*(\b\W{0,2}\d{1,2}(\w{2})?\W{0,2}\b)"
        )

        ## MM-DD-YYYY MM-DD-YY
        date5 = regex.compile(
            r"\b(0[1-9]|1[0-2])[-\/](0[1-9]|[12]\d|3[01])[-\/](\d{2}|\d{4})\b"
        )
        ## DD/MM/YYYY
        date6 = regex.compile(
            r"\b(0[1-9]|[12]\d|3[01])[-/](0[1-9]|1[0-2])[-/](\d{2}|\d{4})\b"
        )
        ## YYYY/MM/DD
        date7 = regex.compile(
            r"\b(\d{2}|\d{4})[-/](0[1-9]|1[0-2])[-/](0[1-9]|[12]\d|3[01])\b"
        )

        res = date1.sub("<DATE>", file_content)
        res = date2.sub("<DATE>", res)
        res = date3.sub("<DATE>", res)
        res = date4.sub("<DATE>", res)
        res = date5.sub("<DATE>", res)
        res = date6.sub("<DATE>", res)
        res = date7.sub("<DATE>", res)
        return res

    ## Split paragraphs because some paragraphs does not have '.' period
    def split_paragraphs(self, file_content):
        res = regex.split(r"((\r?\n){2,})", file_content)
        paragraphs = list(filter(lambda x: len(regex.findall(r"^\n+$", x)) == 0, res))
        new_paras = []
        for para in paragraphs:
            words = para.strip().split(" ")
            unwanted = regex.findall(r"^\*+.*\*+$", para.strip())
            if len(words) >= 3 and len(unwanted) == 0:
                new_paras.append(para)

        return new_paras

    def replace_spaces(self, file_content):
        return regex.sub(r" +", " ", file_content)

    def replace_special_chars_in_word(self, file_content):
        res = regex.sub(r"\b([a-zA-Z]+)-([a-zA-Z]+)\b", r"\1 \2", file_content)
        res = regex.sub(r"[_—\-^$|\+=`~]+", " ", res)
        res = regex.sub(r"\b([a-zA-Z]+)\'([a-zA-Z])\b", r"\1\2", res)
        return res

    def replace_file_name(self, file_content):
        return regex.sub(r"\b[\w-]+\.[A-Za-z]{3}\b", "<FILE_NAME>", file_content)

    def replace_all_other_numeric(self, file_content):
        return regex.sub(r"\b([\:\.\/\+\-\(\)]?\d+){1,}\b", "<NUM>", file_content)

    def replace_upper_chars(self, file_content):
        return file_content.lower()

    def sentence_breaker(self, paragraphs, remove_puctuations=False, break_quote=False):
        sentences = []
        for para in paragraphs:
            new_para = regex.sub(
                r"(?<!\b\w\.)(?<!\b(mrs.)|(ms.)|(mr.)|(dr.))(?<=\.|\!|\?)\s",
                "<POTENTIAL_BREAKER> ",
                para,
            )

            new_sent = new_para.split("<POTENTIAL_BREAKER>")

            current_sentence = ""

            for sent in new_sent:
                current_sentence = current_sentence + sent
                quotes = len(regex.findall(r"\"", current_sentence))
                if quotes % 2 == 0:
                    current_sentence = regex.sub(
                        r"(\n+)", " ", current_sentence.strip()
                    )
                    current_sentence = self.replace_spaces(current_sentence)
                    sentences.append(current_sentence)
                    current_sentence = ""

            if current_sentence != "":
                current_sentence = regex.sub(r"(\n+)", " ", current_sentence.strip())
                current_sentence = self.replace_spaces(current_sentence)
                sentences.append(current_sentence)
                current_sentence = ""

        new_sentences = []
        for para in sentences:
            new_para = regex.sub(
                r"(?<!\b\w\.)(?<!\b(mrs.)|(ms.)|(mr.)|(dr.))(?<=\.|\!|\?)\"(?!\n)",
                '" <POTENTIAL_BREAKER> ',
                para,
            )

            new_sent = new_para.split("<POTENTIAL_BREAKER>")

            current_sentence = ""

            for sent in new_sent:
                current_sentence = current_sentence + sent
                quotes = len(regex.findall(r"\"", current_sentence))
                if quotes % 2 == 0:
                    current_sentence = regex.sub(
                        r"(\n+)", " ", current_sentence.strip()
                    )
                    current_sentence = self.replace_spaces(current_sentence)
                    new_sentences.append(current_sentence)
                    current_sentence = ""

            if current_sentence != "":
                current_sentence = regex.sub(r"(\n+)", " ", current_sentence.strip())
                current_sentence = self.replace_spaces(current_sentence)
                new_sentences.append(current_sentence)
                current_sentence = ""

        sentences = new_sentences

        if break_quote:
            new_sentences = []
            for para in sentences:
                new_para = regex.sub(
                    r"(?<!\b\w\.)(?<!\b\w{3}\.)(?<!\b\w{2}\.)(?<=\.|\?)\s",
                    "<POTENTIAL_BREAKER> ",
                    para,
                )

                new_sent = new_para.split("<POTENTIAL_BREAKER>")

                current_sentence = ""

                for sent in new_sent:
                    current_sentence = regex.sub(r"(\n+)", " ", sent.strip())
                    current_sentence = self.replace_spaces(current_sentence)
                    new_sentences.append(current_sentence)
                    current_sentence = ""

            sentences = new_sentences

        if remove_puctuations:
            new_sentences = []
            for para in sentences:
                new_para = regex.sub(r"\p{P}", " ", para)
                new_sentences.append(self.replace_spaces(new_para))
            sentences = new_sentences

        return sentences

    def word_tokenize(self, sentences):
        result = []

        for sent in sentences:
            new_sent = regex.sub(r"(\p{P})", r" \1 ", sent)
            new_sent = self.replace_spaces(new_sent)
            res = list(filter(lambda x: x != "", new_sent.strip().split(" ")))
            if len(res) > 2:
                result.append(res)

        return result

    def tokenize_content(self, content, remove_puctuations=False, break_quote=False):
        content = self.replace_url(content)
        content = self.replace_email(content)
        content = self.replace_file_name(content)
        content = self.replace_date(content)
        content = self.replace_hashtags(content)
        content = self.replace_mensions(content)
        content = self.replace_all_other_numeric(content)
        content = self.replace_upper_chars(content)
        content = self.replace_spaces(content)
        content = self.replace_special_chars_in_word(content)

        # print(content)

        paras = self.split_paragraphs(content)
        sentences = self.sentence_breaker(paras, remove_puctuations, break_quote)
        words = self.word_tokenize(sentences)

        return words

    def tokenize_doc(self, path, remove_puctuations=False, break_quote=False):
        with open(path, "r") as f:
            file_content = f.read()

        return self.tokenize_content(file_content, remove_puctuations, break_quote)


if __name__ == "__main__":
    file_content = input("Your text: ")

    tokenizer = Tokenizer()
    print(tokenizer.tokenize_content(file_content, break_quote=True))
