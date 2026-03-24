# python new_app/app.py
# source .env/bin/activate
from functools import lru_cache
from pathlib import Path
import sys
import traceback
from flask import Flask, render_template, request, redirect, url_for, abort

# ---------- поиск корня проекта ----------
CURRENT_FILE = Path(__file__).resolve()

def find_project_root(start_path: Path) -> Path:
    for path in [start_path.parent, *start_path.parents]:
        if (path / "src").exists():
            return path
    return start_path.parent

PROJECT_ROOT = find_project_root(CURRENT_FILE)

# чтобы import src работал, даже если app.py лежит не в корне
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import AppContainer  # noqa: E402


# ---------- flask ----------
app = Flask(__name__)

# Создаём контейнер один раз при запуске приложения
container = AppContainer.from_root(
    root_path=PROJECT_ROOT,
    ensure_indexes_on_start=True,
    rebuild_indexes_on_start=False,
)



# ---------- helpers ----------
def parse_int(value, default=None):
    if value in (None, ""):
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_bool(value):
    if value is None or value == "":
        return None
    value = value.lower().strip()
    if value in {"true", "1", "yes"}:
        return True
    if value in {"false", "0", "no"}:
        return False
    return None


def build_search_filters(args):
    filters = {}

    publication_year_from = parse_int(args.get("publication_year_from"))
    publication_year_to = parse_int(args.get("publication_year_to"))
    cited_by_count_min = parse_int(args.get("cited_by_count_min"))
    cited_by_count_max = parse_int(args.get("cited_by_count_max"))

    journal_contains = (args.get("journal_contains") or "").strip()
    field_in_raw = (args.get("field_in") or "").strip()

    if publication_year_from is not None:
        filters["publication_year_from"] = publication_year_from
    if publication_year_to is not None:
        filters["publication_year_to"] = publication_year_to
    if cited_by_count_min is not None:
        filters["cited_by_count_min"] = cited_by_count_min
    if cited_by_count_max is not None:
        filters["cited_by_count_max"] = cited_by_count_max
    if journal_contains:
        filters["journal_contains"] = journal_contains
    if field_in_raw:
        filters["field_in"] = [item.strip() for item in field_in_raw.split(",") if item.strip()]

    print(filters)
    return filters or None


# ---------- routes ----------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/search")
def search():
    available_indexes = container.search_engine_factory.available_indexes()
    return render_template(
        "search.html",
        available_indexes=available_indexes,
    )


@app.route("/results", methods=["GET"])
def results():
    query = (request.args.get("query") or "").strip()
    index_name = (request.args.get("index_name") or "bm25").strip()
    top_k = parse_int(request.args.get("top_k"), 10)
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
        if filters:
            results_data = container.search_service.search_with_filters(
                query=query,
                index_name=index_name,
                top_k=top_k,
                filters=filters,
            )
        else:
            results_data = container.search_service.search(
                query=query,
                index_name=index_name,
                top_k=top_k,
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
    error = None

    if request.method == "POST":
        title = (request.form.get("title") or "").strip()
        description = (request.form.get("description") or "").strip()
        keywords_raw = (request.form.get("keywords") or "").strip()

        keywords = [kw.strip() for kw in keywords_raw.split(",") if kw.strip()]

        if not title:
            error = "Collection title is required."
        else:
            try:
                new_collection = container.collection_manager.create_collection(
                    title=title,
                    description=description,
                    keywords=keywords,
                    autosave=True,
                )
                return redirect(url_for("collection_detail", collection_id=new_collection.id))
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
    try:
        collection = container.collection_manager.get_collection(collection_id)
    except Exception:
        abort(404)

    return render_template(
        "collection_detail.html",
        collection=collection.to_dict(),
    )


@app.route("/collections/<collection_id>/delete", methods=["POST"])
def delete_collection(collection_id):
    try:
        container.collection_manager.delete_collection(collection_id)
    except Exception as exc:
        return render_template(
            "collection_detail.html",
            collection=None,
            error=str(exc),
        ), 400

    return redirect(url_for("collections"))


@app.route("/collections/<collection_id>/remove-paper", methods=["POST"])
def remove_paper_from_collection(collection_id):
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
    collection_id = (request.form.get("collection_id") or "").strip()
    paper_id = (request.form.get("paper_id") or "").strip()
    next_url = (request.form.get("next") or request.referrer or url_for("collections")).strip()

    if not collection_id or not paper_id:
        sep = "&" if "?" in next_url else "?"
        return redirect(f"{next_url}{sep}add_error={quote_plus('Missing collection_id or paper_id')}")

    try:
        added = container.collection_manager.add_paper(
            collection_id=collection_id,
            paper_id=paper_id,
            autosave=True,
        )
        status = "added" if added else "already_exists"
        sep = "&" if "?" in next_url else "?"
        return redirect(f"{next_url}{sep}add_status={status}")

    except Exception as exc:
        sep = "&" if "?" in next_url else "?"
        return redirect(f"{next_url}{sep}add_error={quote_plus(str(exc))}")


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)