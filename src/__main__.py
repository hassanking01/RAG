import json
from pathlib import Path

import fire
from .Rag import RAG


if "__main__" == __name__:
    try:
        result = fire.Fire(RAG)
    except KeyboardInterrupt:
        print("bye")
    # except BaseException:
    #     print("something really bad happened")
