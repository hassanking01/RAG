if "__main__" == __name__:

    try:
        import fire
        from .Rag import RAG
        from rich import print
        from .errors import MaxChunkSizeError, Patherror, EmptyData, EvaluationError
        rag = RAG()

        commands = {
            "index": rag.index,
            "search": rag.search,
            "search_dataset": rag.search_dataset,
            "answer": rag.answer,
            "answer_dataset": rag.answer_dataset,
            "evaluate": rag.evaluate,
        }

        fire.Fire(commands)

    except KeyboardInterrupt:
        print("good bye")
    except (MaxChunkSizeError, Patherror, EmptyData, EvaluationError, ValueError) as e:
        print(
            f"[red]{e.__class__.__name__}[/red]:",
            e)
