import argparse
import sys
from pathlib import Path
from src.app.container import AppContainer
from pprint import pprint
import shlex


class CLIApplication:
    """
    Class to render CLI logic and execution
    """

    def __init__(self):
        """
        Initialise CLI app with argument parser setup
        """
        # main ArgumentParser class that processes commands in CLI
        self.parser = argparse.ArgumentParser(description="SciTriveo: Research copilot")

        # AppContainer instance that helps with commands
        self.container = None

        # setup all arguments and subparsers
        self._setup_parser()

    def _setup_parser(self):
        """
        Internal function to configure the argument parser with subcommands for research operations
        """
        # add argument for path
        self.parser.add_argument("--root", type=str, default=".", help="path to project root")

        # add subparsers for commands like search, collections recommend
        subparsers = self.parser.add_subparsers(dest="command", help="available service commands")

        # search subcommands
        search_ptr = subparsers.add_parser("search", help="perform document search")
        search_ptr.add_argument("query", type=str, help="search query string")
        search_ptr.add_argument("--index", type=str, default="bm25", choices=["bm25", "word2vec", "fasttext"])
        search_ptr.add_argument("--top-k", type=int, default=5)

        # collection subcommand group
        coll_ptr = subparsers.add_parser("collection", help="manage user collections")
        coll_sub = coll_ptr.add_subparsers(dest="subcommand")

        # collection create
        create_ptr = coll_sub.add_parser("create", help="initialise a new collection")
        create_ptr.add_argument("--title", required=True, type=str)
        create_ptr.add_argument("--desc", type=str, default="")
        create_ptr.add_argument("--keywords", nargs="+", default=[])

        # collection add
        add_ptr = coll_sub.add_parser("add", help="add paper to a collection")
        add_ptr.add_argument("--id", required=True, help="target collection uuid")
        add_ptr.add_argument("--paper-id", required=True, help="stable paper identifier")

        # collection list
        list_ptr = coll_sub.add_parser("list", help="list user collections")
        list_ptr.add_argument("--show-papers", action="store_true", help="display detailed paper metadata in list")

        # recommend subcommand
        rec_ptr = subparsers.add_parser("recommend", help="generate recommendations for a collection")
        rec_ptr.add_argument("--id", required=True, help="source collection uuid")
        rec_ptr.add_argument("--top-k", type=int, default=5)

        # interactive mode command
        subparsers.add_parser("interactive", help="enter persistent session to avoid reloads")

        # rag command
        # configuration for the rag assistant
        ask_ptr = subparsers.add_parser("ask", help="ask a question about a collection")
        ask_ptr.add_argument("--id", required=True, help="collection identifier")
        ask_ptr.add_argument("question", type=str, help="question text")

    @staticmethod
    def _print_formatted_hits(hits):
        """
        Method to format search or recommendation results in table
        :param hits: list of result dictionaries containing scores and text
        """
        if not hits:
            print("\nno records found")
            return

        # dynamic template for metadata display
        line_format = "{:<12} | {:<6} | {:<4} | {:<4} | {:<25} | {:<40}"
        header = line_format.format("Paper ID", "Score", "Year", "Cite", "Journal", "Title")
        print("\n" + header)
        print("-" * len(header))

        for hit in hits:
            title = str(hit.get("title") or "no title")
            journal = str(hit.get("journal") or "unknown")
            year = str(hit.get("publication_year") or "n/a")
            cites = str(hit.get("cited_by_count") or "zero")

            print(line_format.format(
                str(hit["paper_id"])[:12],
                f"{hit['score']:.4f}",
                year[:4],
                cites[:4],
                journal[:25],
                title[:40]
            ))
        print("-" * len(header))

    def _initialise_container(self, root_path: str):
        """
        Launch heavy application container if not already present
        :param root_path: path to the project data root
        """
        if self.container is None:
            try:
                self.container = AppContainer.from_root(
                    root_path=Path(root_path),
                    ensure_indexes_on_start=True
                )
            except Exception as error:
                print(f"fatal initialization error: {error}")
                sys.exit(1)

    def run(self):
        """
        Main function to parse arguments and direct execution to service methods
        """
        # parse the initial command line arguments
        args = self.parser.parse_args()

        if args.command == "interactive":
            # start running session to keep services in memory
            self._run_interactive_shell(args.root)
        elif args.command:
            # perform one-off command execution
            self._initialise_container(args.root)
            self._execute_command(args)
        else:
            # print help menu if no input is provided
            self.parser.print_help()

    def _run_interactive_shell(self, root: str):
        """
        Infinite loop to process commands without reloading app container
        :param root: project root for container initialization
        """
        print("Loading app for interactive session...")
        self._initialise_container(root)
        print("App is ready. type 'exit' or 'quit' to close the session")

        while True:
            try:
                user_input = input("scitriveo > ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ("exit", "quit"):
                    break

                # split input considering quoted strings as single arguments
                cmd_args = shlex.split(user_input)
                parsed = self.parser.parse_args(cmd_args)
                self._detect_command(parsed)
            except SystemExit:
                # catch argparse attempts to exit on help or error
                continue
            except KeyboardInterrupt:
                print("\nSession is terminated by user")
                break
            except Exception as error:
                print(f"Execution error: {error}")

    def _detect_command(self, args: argparse.Namespace):
        """
        Internal method to detect command arguments
        :param args: parsed namespace of arguments
        """
        if args.command == "search":
            self._handle_search(args)
        elif args.command == "collection":
            self._handle_collection(args)
        elif args.command == "recommend":
            self._handle_recommend(args)
        elif args.command == "interactive":
            print("already in interactive mode")
        elif args.command == "ask":
            self._handle_ask(args)
        else:
            self.parser.print_help()

    def _handle_search(self, args: argparse.Namespace):
        """
        Internal function to execute search logic and display results
        :param args: parsed command line arguments
        """
        print(f"Executing search for: {args.query}")
        response = self.container.search(args.query, args.index, top_k=args.top_k)

        # display each result block similarly to the internal service output
        for hit in response["results"]:
            print("\n--- Document found ---")
            pprint(hit, indent=2, width=100)

        print(f"\nLatency: {response['elapsed_ms']} ms | total found: {response['results_count']}")

    def _handle_collection(self, args: argparse.Namespace):
        """
        Internal method to connect collection logic
        :param args: command line arguments for collections
        """
        if args.subcommand == "create":
            collection = self.container.create_collection(args.title, args.desc, args.keywords)
            print("Collection created:")
            pprint(collection)

        elif args.subcommand == "add":
            success = self.container.add_paper_to_collection(args.id, args.paper_id)
            if success:
                print("Paper successfully added to collection")
            else:
                print("Failed to update collection papers")

        elif args.subcommand == "list":
            collections = self.container.collection_manager.list_collections()
            for item in collections:
                print(f"\nCollection: {item.title} (id: {item.id})")
                print(f"Description: {item.description}")
                print(f"Keywords: {item.keywords}")
                print(f"Papers count: {len(item.added_papers)}")

                if args.show_papers and item.added_papers:
                    # extract full metadata rows for membership display
                    print("Added papers details:")
                    mask = self.container.docs_df['id'].isin(item.added_papers)
                    subset = self.container.docs_df[mask]
                    for _, row in subset.iterrows():
                        # print a compact version of the record
                        print(f"  - [{row['id']}] {row['title'][:70]}...")

    def _handle_recommend(self, args: argparse.Namespace):
        """
        Internal method to launch recommendation engine for the collection
        :param args: command line arguments for recommendation
        """
        print(f"Generating recommendations for collection: {args.id}")
        response = self.container.recommend(args.id, top_k=args.top_k)

        for rec in response["results"]:
            print("\n--- Recommended document ---")
            pprint(rec, indent=2, width=100)

        print(f"\nMode: {response['mode']} | latency: {response['elapsed_ms']} ms")

    def _handle_ask(self, args: argparse.Namespace):
        """
        Internal function to execute generative answering logic
        :param args: parsed command line arguments
        """
        print(f"Consulting research assistant for collection: {args.id}")
        # this will trigger model loading on the first call
        answer = self.container.ask_collection(args.id, args.question)
        print("\n--- Assistant Response ---")
        print(answer)
        print("---------------------------")
