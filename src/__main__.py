if "__main__" == __name__:
    try:
        import fire
        from .Rag import RAG
        result = fire.Fire(RAG)
    except KeyboardInterrupt:
        print("bye")
    except BaseException:
        print("something really bad happened")
