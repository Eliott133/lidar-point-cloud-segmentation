import typer

from hub.services.registry import get_adapter

app = typer.Typer()


@app.command()
def train(model: str):
    adapter = get_adapter(model)
    result = adapter.train({})
    typer.echo(result)


@app.command()
def evaluate(model: str):
    adapter = get_adapter(model)
    result = adapter.evaluate({})
    typer.echo(result)


if __name__ == "__main__":
    app()