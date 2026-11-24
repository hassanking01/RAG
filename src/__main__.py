if "__main__" == __name__:
    try:
        import fire
        from .Rag import RAG
        from rich import print, traceback
        from .errors import MaxChunkSizeError, Patherror, EmptyData, EvaluationError
        traceback.install()       
        result = fire.Fire(RAG)
    except KeyboardInterrupt:
        print("good bye")
    except (MaxChunkSizeError, Patherror, EmptyData, EvaluationError, ValueError) as e:
        print(
            f"[red]{e.__class__.__name__}[/red]:",
            e)
