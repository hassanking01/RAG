if "__main__" == __name__:
    try:
        import fire
        from .Rag import RAG
        from rich import print, traceback
        from .errors import MaxChunkSizeError, Patherror, EmptyData
        traceback.install()       
        result = fire.Fire(RAG)
    except KeyboardInterrupt:
        print("bye")
    except (MaxChunkSizeError, Patherror, EmptyData) as e:
        print(
            f"[red]{e.__class__.__name__}[/red]:",
            e)
    except BaseException as e:
        print(f"something really bad happened {e}")
