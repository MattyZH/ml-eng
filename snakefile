rule plot:
    input:
        cancer = "data/cancer.csv",
        spam = "data/spam.csv"
    output:
        cancer = "./graphs/cancer",
        spam = "./graphs/spam"
    script:
        "./script/t_script.py"