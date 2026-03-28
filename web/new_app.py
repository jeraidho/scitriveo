from pathlib import Path
import sys
import traceback
from urllib.parse import quote_plus
from flask import Flask, render_template, request, redirect, url_for, abort

# identify the directory structure for relative imports
current_file = Path(__file__).resolve()


def find_project_root(start_path: Path) -> Path:
    """
    traverse directory tree to locate the project root folder

    :param start_path: initial path for the search
    :returns: path to the project root
    """
    for path in [start_path.parent, *start_path.parents]:
        if (path / "src").exists():
            return path
    return start_path.parent


project_root = find_project_root(current_file)

# ensure the root directory is in the system path for module loading
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src import AppContainer

# initialise the flask application instance
app = Flask(__name__)

# instantiate the application container once at startup
# this loads the search indices and heavy static models into memory
container = AppContainer.from_root(
    root_path=project_root,
    ensure_indexes_on_start=True,
    rebuild_indexes_on_start=False,
)


def parse_integer(value, default=None):
    """
    utility to convert string input into an integer with a fallback

    :param value: input string from request arguments
    :param default: value to return if conversion fails
    :returns: converted integer or default value
    """
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def build_search_filters(args):
    """
    map request arguments to the internal search filter schema

    :param args: immutable dictionary of request parameters
    :returns: dictionary of filters or none if empty
    """
    filters = {}

    # extract numerical constraints
    year_min = parse_integer(args.get("publication_year_from"))
    year_max = parse_integer(args.get("publication_year_to"))
    cite_min = parse_integer(args.get("cited_by_count_min"))
    cite_max = parse_integer(args.get("cited_by_count_max"))

    # extract string matching constraints
    journal = (args.get("journal_contains") or "").strip()
    fields_raw = (args.get("field_in") or "").strip()

    if year_min is not None:
        filters["publication_year_from"] = year_min
    if year_max is not None:
        filters["publication_year_to"] = year_max
    if cite_min is not None:
        filters["cited_by_count_min"] = cite_min
    if cite_max is not None:
        filters["cited_by_count_max"] = cite_max
    if journal:
        filters["journal_contains"] = journal
    if fields_raw:
        filters["field_in"] = [item.strip() for item in fields_raw.split(",") if item.strip()]

    return filters or None


@app.route("/")
def index():
    """
    render the main landing page with project description
    """
    return render_template("index.html")


@app.route("/search")
def search():
    """
    render the search configuration page with index options
    """
    available_indexes = container.search_engine_factory.available_indexes()
    return render_template(
        "search.html",
        available_indexes=available_indexes,
    )


@app.route("/results", methods=["GET"])
def results():
    """
    execute search logic and render the findings page
    """
    query = (request.args.get("query") or "").strip()
    index_name = (request.args.get("index_name") or "bm25").strip()
    top_k = parse_integer(request.args.get("top_k"), 10)
    filters = build_search_filters(request.args)

    collections_data = container.collection_manager.to_list_of_dicts()
    add_status = request.args.get("add_status")
    add_error = request.args.get("add_error")

    if not query:
        return render_template(
            "results.html",
            query=query,
            index_name=index_name,
            top_k=top_k,
            filters=filters or {},
            results_data=None,
            error=None,
            collections=collections_data,
            add_status=add_status,
            add_error=add_error,
        )

    try:
        # use the container facade for filtered or standard search
        results_data = container.search(
            query=query,
            index_name=index_name,
            top_k=top_k,
            filters=filters
        )

        return render_template(
            "results.html",
            query=query,
            index_name=index_name,
            top_k=top_k,
            filters=filters or {},
            results_data=results_data,
            error=None,
            collections=collections_data,
            add_status=add_status,
            add_error=add_error,
        )

    except Exception as exc:
        traceback.print_exc()
        return render_template(
            "results.html",
            query=query,
            index_name=index_name,
            top_k=top_k,
            filters=filters or {},
            results_data=None,
            error=repr(exc),
            collections=collections_data,
            add_status=add_status,
            add_error=add_error,
        )


@app.route("/collections", methods=["GET", "POST"])
def collections():
    """
    handle the creation and listing of research collections
    """
    error = None

    if request.method == "POST":
        title = (request.form.get("title") or "").strip()
        description = (request.form.get("description") or "").strip()
        keywords_raw = (request.form.get("keywords") or "").strip()

        keywords = [kw.strip() for kw in keywords_raw.split(",") if kw.strip()]

        if not title:
            error = "collection title is required"
        else:
            try:
                new_coll = container.collection_manager.create_collection(
                    title=title,
                    description=description,
                    keywords=keywords,
                    autosave=True,
                )
                return redirect(url_for("collection_detail", collection_id=new_coll.id))
            except Exception as exc:
                error = str(exc)

    collections_data = container.collection_manager.to_list_of_dicts()
    return render_template(
        "collections.html",
        collections=collections_data,
        error=error,
    )


@app.route("/collections/<collection_id>")
def collection_detail(collection_id):
    """
    display detailed information about a specific collection and its papers
    """
    try:
        collection = container.collection_manager.get_collection(collection_id)
    except Exception:
        abort(404)

    # fetch full metadata for all papers linked to this collection
    paper_metadata = []
    if collection.added_papers:
        mask = container.docs_df['id'].isin(collection.added_papers)
        paper_metadata = container.docs_df[mask].to_dict('records')

    return render_template(
        "collection_detail.html",
        collection=collection.to_dict(),
        papers=paper_metadata
    )


@app.route("/collections/<collection_id>/recommend")
def collection_recommend(collection_id):
    """
    generate and display papers related to the collection profile
    """
    try:
        top_k = parse_integer(request.args.get("top_k"), 10)
        # invoke the recommendation service via the container facade
        recommendations = container.recommend(collection_id, top_k=top_k)

        collection = container.collection_manager.get_collection(collection_id)

        return render_template(
            "recommendations.html",
            collection=collection.to_dict(),
            recommendations=recommendations
        )
    except Exception as exc:
        traceback.print_exc()
        return redirect(url_for("collection_detail", collection_id=collection_id))


@app.route("/collections/<collection_id>/delete", methods=["POST"])
def delete_collection(collection_id):
    """
    permanently remove a collection from the system
    """
    try:
        container.collection_manager.delete_collection(collection_id)
    except Exception as exc:
        abort(400)

    return redirect(url_for("collections"))


@app.route("/collections/<collection_id>/remove-paper", methods=["POST"])
def remove_paper_from_collection(collection_id):
    """
    unlink a specific paper from a collection
    """
    paper_id = (request.form.get("paper_id") or "").strip()

    if paper_id:
        try:
            container.collection_manager.remove_paper(
                collection_id=collection_id,
                paper_id=paper_id,
                autosave=True,
            )
        except Exception:
            abort(400)

    return redirect(url_for("collection_detail", collection_id=collection_id))


@app.route("/collections/add-paper", methods=["POST"])
def add_paper_to_collection_from_results():
    """
    add a paper found in search results to a user collection
    """
    collection_id = (request.form.get("collection_id") or "").strip()
    paper_id = (request.form.get("paper_id") or "").strip()
    next_url = (request.form.get("next") or request.referrer or url_for("collections")).strip()

    if not collection_id or not paper_id:
        return redirect(f"{next_url}&add_error={quote_plus('missing parameters')}")

    try:
        added = container.collection_manager.add_paper(
            collection_id=collection_id,
            paper_id=paper_id,
            autosave=True,
        )
        status = "added" if added else "already_exists"
        return redirect(f"{next_url}&add_status={status}")

    except Exception as exc:
        return redirect(f"{next_url}&add_error={quote_plus(str(exc))}")


# @app.route("/collections/<collection_id>/ask", methods=["GET", "POST"])
# def collection_ask(collection_id):
#     """
#     handle generative question answering for a specific collection
#
#     :param collection_id: unique identifier of the collection
#     :returns: rendered assistant page with answer if post request
#     """
#     answer = None
#     question = None
#
#     try:
#         # retrieve collection to ensure it exists and for metadata display
#         collection = container.collection_manager.get_collection(collection_id)
#     except Exception:
#         abort(404)
#
#     if request.method == "POST":
#         question = request.form.get("question", "").strip()
#         if question:
#             try:
#                 # call the container facade which orchestrates rag logic
#                 answer = container.ask_collection(collection_id, question)
#             except Exception as exc:
#                 traceback.print_exc()
#                 answer = f"error during synthesis: {str(exc)}"
#
#     return render_template(
#         "ask_assistant.html",
#         collection=collection.to_dict(),
#         question=question,
#         answer=answer
#     )

@app.route("/collections/<collection_id>/ask", methods=["GET", "POST"])
def collection_ask(collection_id):
    """
    handle generative question answering with support for asynchronous updates

    :param collection_id: unique identifier of the collection
    :returns: rendered html for initial load or json for background requests
    """
    try:
        # verify collection existence
        collection = container.collection_manager.get_collection(collection_id)
    except Exception:
        abort(404)

    # check if request is made via background fetch
    is_ajax = request.headers.get('X-Requested-With') == 'XMLHttpRequest'

    if request.method == "POST":
        question = request.form.get("question", "").strip()
        if not question:
            return {"error": "empty question"}, 400 if is_ajax else redirect(
                url_for('collection_ask', collection_id=collection_id))

        try:
            # perform synthesis using the rag service
            answer = container.ask_collection(collection_id, question)

            if is_ajax:
                return {"answer": answer, "question": question}, 200

            return render_template(
                "ask_assistant.html",
                collection=collection.to_dict(),
                question=question,
                answer=answer
            )
        except Exception as exc:
            traceback.print_exc()
            if is_ajax:
                return {"error": str(exc)}, 500
            return render_template("ask_assistant.html", collection=collection.to_dict(), error=str(exc))

    # default get request
    return render_template("ask_assistant.html", collection=collection.to_dict())

if __name__ == "__main__":
    # disable reloader to prevent multiple heavy model loads
    app.run(debug=True, use_reloader=False)
