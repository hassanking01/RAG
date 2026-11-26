if "__main__" == __name__:

    try:
        import fire
        from .Rag import RAG
        from rich import print
        from sys import argv
        from .errors import (
            MaxChunkSizeError,
            Patherror,
            EmptyData,
            EvaluationError,
        )
        rag = RAG()

        commands = {
            "index": rag.index,
            "search": rag.search,
            "search_dataset": rag.search_dataset,
            "answer": rag.answer,
            "answer_dataset": rag.answer_dataset,
            "evaluate": rag.evaluate,
        }

        dict_funcs = dir(commands)
        not_allowed = list(arg for arg in argv if arg in dict_funcs)
        if not_allowed:
            print(
                "the following arguments are not part of this cli: "
                f"{not_allowed}"
            )
            print(f"this is the only funtions allowd {list(commands.keys())}")
            exit()

        result = fire.Fire(commands, serialize=lambda _: None)
        if result:
            print(result)

    except KeyboardInterrupt:
        print("good bye")
    except (
        MaxChunkSizeError,
        Patherror,
        EmptyData,
        EvaluationError,
        ValueError,
        IndexError,
    ) as e:
        print(f"[red]{e.__class__.__name__}[/red]:", e)
