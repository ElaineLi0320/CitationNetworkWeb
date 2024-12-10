import pandas as pd


def read_paper_txt(file_path):
    """
    This function is used to convert the original dataset into dataFrame, 
    and filter the paper without reference.

    The data schema of the original dataset:
    #* --- paperTitle
    #@ --- Authors
    #t --- Year
    #c --- publication venue
    #index --- index id of this paper
    #% --- the id of references of this paper 
        (there are multiple lines, with each indicating a reference)
    #! --- Abstract

    After the preprocess each paper:
    {'title': 'Becoming a virtual organism to learn about genetics', 
     'authors': 'Alexander Bick', 
     'year': '2006', 
     'venue': 'Crossroads', 
     'index': '198', 
     'references': ['89531']}
    """
    papers = []
    current_paper = {}
    references = []

    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()

                #if parse another "#*", means need to store the previous paper
                if line.startswith('#*') and current_paper:
                    # filter add reference info
                    # if references and ('authors' in current_paper) and current_paper['authors'].strip():
                    #     current_paper['references'] = references
                    #     papers.append(current_paper.copy())

                    #try if not filter
                    if references:
                        current_paper['references'] = references.copy()
                    papers.append(current_paper.copy())
                    current_paper = {}
                    references = []

                if line.startswith('#*'):
                    current_paper['title'] = line[2:].strip()
                elif line.startswith('#@'):
                    current_paper['authors'] = line[2:].strip()
                elif line.startswith('#t'):
                    current_paper['year'] = line[2:].strip()
                elif line.startswith('#c'):
                    current_paper['venue'] = line[2:].strip()
                elif line.startswith('#index'):
                    current_paper['index'] = line[6:].strip()
                elif line.startswith('#%'):
                    references.append(line[2:].strip())
                elif line.startswith('#!'):
                    current_paper['abstract'] = line[2:].strip()
            
        if current_paper:
        # if current_paper and references and ('authors' in current_paper) and current_paper['authors'].strip():
            if references:
                current_paper['references'] = references.copy()
            papers.append(current_paper.copy())

        # print(papers)

        df = pd.DataFrame(papers)
        # print(f"The number of papers with reference: {total_papers}")

        return df

    except Exception as e:
        print(f"Error of reading the file: {str(e)}")
        return None
    
def export_as_test(df, output_path):
    """
    This function is used to export the dataset for checking
    """
    try:
        df.to_csv(output_path, index=False)
        print(f"export Successfully: {output_path}")
    except Exception as e:
        print(f"Error of exporting the file: {str(e)}")


def generate_simple_report(df):
    """
    Generate a simple data quality report
    """
    print("\nData Quality Report:")
    print("===================")
    print(f"Total Records: {len(df)}")
    print("\nColumn Analysis:")
    
    for column in df.columns:
        # Calculate basic metrics
        null_count = df[column].isna().sum()
        empty_string_count = len(df[df[column] == '']) if df[column].dtype == 'object' else 0
        valid_count = len(df) - null_count - empty_string_count
        
        print(f"\n{column}:")
        print(f"  Null values: {null_count}")
        print(f"  Empty strings: {empty_string_count}")
        print(f"  Valid values: {valid_count}")
        
def filter_paper(df):
    """
    To keep papers that:
        1. have reference list AND
        2. exist in other paper's reference list
    """
    cited_paper_id = set()
    for _, paper in df.iterrows():
        if isinstance(paper['references'], list):
            cited_paper_id.update(str(ref) for ref in paper['references'])

    filtered_df = df[df['index'].astype(str).isin(cited_paper_id)]

    papers_to_keep = []
    for _, paper in df.iterrows():
        paper_id = str(paper['index'])
        has_references = isinstance(paper['references'], list) and len(paper['references']) > 0
        is_cited = paper_id in cited_paper_id
        
        if has_references and is_cited:
            papers_to_keep.append(paper_id)

    filtered_df = df[df['index'].astype(str).isin(papers_to_keep)]

    print(f"\nFiltering Statistics:")
    print(f"Original papers: {len(df)}")
    print(f"Papers after filtering: {len(filtered_df)}")
    print(f"Removed papers: {len(df) - len(filtered_df)}")
    print(f"Papers being cited: {len(cited_paper_id)}")
    
    return filtered_df
        

# file_path = 'outputacm.txt'
# output_excel_path = 'test.csv' 
# df = read_paper_txt(file_path)

# sampled_df = df.head(100000)

# filtered_df = filter_paper(sampled_df)
# generate_simple_report(filtered_df)

# export_as_test(filtered_df, output_excel_path)
