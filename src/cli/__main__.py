from src.cli.cli_app import CLIApplication


def launch():
    """
    Entry point function to run the cli application
    """
    # create the application instance
    app = CLIApplication()
    # start the execution loop
    app.run()


if __name__ == "__main__":
    launch()
