from somd2.config import Config


def configure(inp):
    conf = Config(**inp)
    return conf


# c = configure({"pressure": "1 bar", "minimise": True, "extra_args": {"a": 1}})
# print(c.equilibrate)


cnf = Config()
