import logging
from typing import Dict, List, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def preprocess_gtr_persons(gtr_persons: pd.DataFrame) -> pd.DataFrame:
    """Preprocess GtR persons data.

    Args:
        gtr_persons (pd.DataFrame): DataFrame containing GtR person information with columns:
            - projects: List of project dictionaries containing project IDs
            - organisations: List of organisation dictionaries containing org IDs
            - orcid_id: ORCID identifier
            - other_names: Alternative names (dropped)
            - email: Email address (dropped)

    Returns:
        pd.DataFrame: Preprocessed DataFrame with:
            - projects: List of project IDs
            - organisation: Single organisation ID (first one)
            - orcid_gtr: Renamed from orcid_id
            - other_names and email columns dropped
    """
    # preprocess gtr persons data
    gtr_persons["projects"] = gtr_persons["projects"].apply(
        lambda x: [project["id"] for project in x]
    )
    gtr_persons["organisations"] = gtr_persons["organisations"].apply(
        lambda x: [organisation["id"] for organisation in x][0]
    )
    gtr_persons.rename(
        columns={"orcid_id": "orcid_gtr", "organisations": "organisation"}, inplace=True
    )
    gtr_persons.drop(columns=["other_names", "email"], inplace=True)

    return gtr_persons


def preprocess_gtr_projects(gtr_projects: pd.DataFrame) -> Dict[str, List[str]]:
    """Preprocess GtR projects data.

    Args:
        gtr_projects (pd.DataFrame): DataFrame containing GtR project information with columns:
            - project_id: Unique identifier for the project
            - publications: List of publication identifiers
            - start_date: Project start date
            - end_date: Project end date
            - extended_end: Extended project end date (optional)

    Returns:
        Dict[str, List[str]]: Dictionary mapping project IDs to lists containing:
            - List of publication IDs
            - Project start date
            - Project end date (using extended_end if available)
    """
    # extract the publications (list, split in "/", take last)
    gtr_projects["publications"] = gtr_projects["publications"].apply(
        lambda x: [publication.split("/")[-1] for publication in x]
    )

    # if extended_end is None, set it to end_date
    gtr_projects["extended_end"] = gtr_projects["extended_end"].fillna(
        gtr_projects["end_date"]
    )
    gtr_projects["end_date"] = gtr_projects["extended_end"]

    # create a single list per project with the list of publications
    gtr_projects["project_info"] = gtr_projects.apply(
        lambda row: [row["publications"], row["start_date"], row["end_date"]], axis=1
    )

    # return a dictionary of project_id as key and project_info as value
    return gtr_projects.set_index("project_id")["project_info"].to_dict()


def preprocess_gtr_topics(
    gtr_project_topics: pd.DataFrame, cwts_taxonomy: pd.DataFrame
) -> Dict[str, List[tuple]]:
    """Preprocess GtR project topics and match with CWTS taxonomy.

    Args:
        gtr_project_topics: DataFrame containing GtR project topics
        cwts_taxonomy: DataFrame containing CWTS taxonomy information

    Returns:
        Dictionary mapping project IDs to their topic information as tuples.
        Each tuple contains (topic_id, topic_name, subfield_id, subfield_name,
        field_id, field_name, domain_id, domain_name). Lower-level fields may
        be None for partial matches.
    """
    # Initial preprocessing
    topics = gtr_project_topics.copy()

    # filter by confidence
    filter_scores = ["medium", "high", "very high"]
    topics = topics[topics["zeroshot_favouring_confidence"].isin(filter_scores)]
    topics.rename(columns={"zeroshot_favouring_confidence": "confidence"}, inplace=True)

    # select relevant columns and split taxonomy label
    topics = topics[["project_id", "taxonomy_label", "confidence"]]
    topics["topic_hierarchy"] = topics["taxonomy_label"].str.split(" > ")

    # left merge topics with cwts_taxonomy
    cwts_taxonomy = cwts_taxonomy.rename(columns={"label": "taxonomy_label"})
    topics = topics.merge(
        cwts_taxonomy[["taxonomy_label", "id_path"]], on="taxonomy_label", how="left"
    )

    # process topics by project
    result = {}
    for project_id, group in topics.groupby("project_id"):
        topic_lists = []
        for _, row in group.iterrows():
            topic_list = _create_topic_list(row["topic_hierarchy"], row["id_path"])
            if topic_list and any(topic_list):  # include if at least one level matched
                topic_lists.append(topic_list)

        if topic_lists:  # only include valid topics
            result[project_id] = topic_lists

    return result


def preprocess_gtr_publications(
    gtr_publications: pd.DataFrame, oa_publications: pd.DataFrame
) -> Dict[str, List[str]]:
    """Preprocess GtR publications data.

    Args:
        gtr_publications: DataFrame containing GtR publication information
        oa_publications: DataFrame containing OpenAlex publication information

    Returns:
        Dictionary mapping project IDs to lists containing:
            - List of publication IDs
            - List of authors (author_id, count)
    """
    # clean the doi column
    gtr_publications["doi"] = gtr_publications["doi"].str.extract(r"(10\..+)")
    oa_publications["doi"] = oa_publications["doi"].str.extract(r"(10\..+)")

    # drop duplicate doi from oa_publications
    oa_publications = oa_publications.drop_duplicates(subset="doi")

    # Merge publications and extract author info in one step
    merged = gtr_publications.merge(
        oa_publications[["doi", "id", "authorships"]], on="doi", how="inner"
    )

    result = {}
    # Process each project group separately to reduce memory usage
    for project_id, group in merged.groupby("project_id"):
        author_counts = {}
        publication_ids = []

        for _, row in group.iterrows():
            if isinstance(row["authorships"], np.ndarray):
                for author_id in [item[0] for item in row["authorships"]]:
                    author_counts[author_id] = author_counts.get(author_id, 0) + 1
            publication_ids.append(row["id"])

        result[project_id] = {
            "authors": [[author, count] for author, count in author_counts.items()],
            "id": publication_ids,
        }

    return result


def map_project_info(project_id: str, projects_dict: Dict, topics_dict: Dict) -> Dict:
    """Map project ID to its associated information and topics.

    Args:
        project_id: The ID of the project
        projects_dict: Dictionary mapping project IDs to project info
        topics_dict: Dictionary mapping project IDs to topic tuples

    Returns:
        Dictionary containing project information or None if project not found
    """
    if project_id in projects_dict:
        return {
            "project_id": project_id,
            "publications": projects_dict[project_id][0],
            "start_date": projects_dict[project_id][1],
            "end_date": projects_dict[project_id][2],
            "topics": topics_dict.get(project_id, []),
        }
    return {
        "project_id": "",
        "publications": [],
        "start_date": "",
        "end_date": "",
        "topics": [],
    }


def flatten_and_aggregate_authors(authors_list: List) -> List[List[str]]:
    """Flatten and aggregate authors from deeply nested lists.

    Args:
        authors_list: List of lists containing author IDs and counts, structured as:
            [
                [[[A1234, 2], [A5678, 1]]],  # First project
                [[[A1234, 1], [A9012, 3]]]   # Second project
            ]
            Can also contain empty lists [[]] or other invalid data

    Returns:
        List of [author_id, total_count] pairs, e.g.:
            [[A1234, 3], [A5678, 1], [A9012, 3]]
    """
    author_counts = {}

    # Handle non-list input
    if not isinstance(authors_list, list):
        return []

    # iterate through the nested structure
    for project_list in authors_list:
        if not isinstance(project_list, list):  # skip non-list items
            continue

        for author_pair in project_list:
            if (
                not isinstance(author_pair, list) or len(author_pair) != 2
            ):  # skip invalid pairs
                continue
            try:
                author_id, count = author_pair
                if isinstance(count, (int, float)):
                    if author_id not in author_counts:
                        author_counts[author_id] = 0
                    author_counts[author_id] += count
            except (ValueError, TypeError):
                continue

    # convert to sorted list of [author_id, total_count] pairs
    return sorted([[str(author_id), str(count)] for author_id, count in author_counts.items()])


def _create_topic_list(hierarchy: List[str], id_path: str) -> Optional[tuple]:
    """Create a standardised topic tuple from hierarchy and ID path.

    Args:
        hierarchy: List of topic names from domain to specific topic
        id_path: String of IDs separated by " > " from domain to topic

    Returns:
        Tuple containing (topic_id, topic_name, subfield_id, subfield_name,
        field_id, field_name, domain_id, domain_name) or None if invalid.
        For partial matches, higher-level categories are preserved and lower
        levels are set to None.
    """
    if not hierarchy or not id_path:
        return None

    ids = id_path.split(" > ")
    if len(hierarchy) != len(ids):
        return None

    # Initialise all values as None
    topic_id, topic_name = None, None
    subfield_id, subfield_name = None, None
    field_id, field_name = None, None
    domain_id, domain_name = None, None

    # fill in values based on available depth, top-down
    depth = len(hierarchy)

    if depth >= 1:  # Domain level
        domain_id = ids[0]
        domain_name = hierarchy[0]

    if depth >= 2:  # Field level
        field_id = ids[1]
        field_name = hierarchy[1]

    if depth >= 3:  # Subfield level
        subfield_id = ids[2]
        subfield_name = hierarchy[2]

    if depth >= 4:  # Topic level
        topic_id = f"T{ids[3]}"
        topic_name = hierarchy[3]

    return [
        topic_id,
        topic_name,
        subfield_id,
        subfield_name,
        field_id,
        field_name,
        domain_id,
        domain_name,
    ]
