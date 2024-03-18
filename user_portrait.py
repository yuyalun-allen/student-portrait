import os
import json
from datetime import datetime
import psycopg2
import pandas as pd
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import linkage
import yaml


class DBConnectionInfo:
    def __init__(self):
        with open("connection.yml", "r") as f:
            config = yaml.load(f, yaml.FullLoader)
            self.host = config["host"]
            self.port = config["port"]
            self.database = config["database"]
            self.user = config["user"]
            self.password = config["password"]


def import_student_answer():
    poll_vote = import_table_to_pd("misago_threads_pollvote")
    poll = import_table_to_pd("misago_threads_poll")
    thread = import_table_to_pd("misago_threads_thread")

    poll_info = pd.merge(thread[["title", "id"]], 
                        poll[["thread_id", "choices"]], 
                        left_on="id",
                        right_on="thread_id").drop(columns=["id"])
    students_answers = pd.merge(poll_info, poll_vote[["thread_id", "choice_hash", "voter_id"]], on="thread_id")

    def match_label(row):
        for item in row['choices']:
            if item['hash'] == row['choice_hash']:
                return item['label']
        return None
    

    students_answers['label'] = students_answers.apply(match_label, axis=1)
    return students_answers


def import_question_answers(path):
    xlsx_files = [file for file in os.listdir(path) if file.endswith('.xlsx')]

    # 创建一个空列表来存储所有DataFrame
    dataframes = []

    # 遍历xlsx文件并导入为DataFrame
    for file in xlsx_files:
        file_path = os.path.join(path, file)
        df = pd.read_excel(file_path)
        df['问题编号'] = file[:-5] + df['问题编号'].astype(str)
        dataframes.append(df)

    # 合并所有DataFrame
    return pd.concat(dataframes, ignore_index=True)


def assess_students(student_answers, question_answers):
    full_df = pd.merge(left=student_answers,
                    right=question_answers,
                    left_on="title",
                    right_on="问题编号")
    
    def judge(row):
        if row["label"][0] in row["答案"]:
            return True
        else:
            return False

    answer_judgements = full_df.loc[:, ["title", "voter_id", "特征"]]
    answer_judgements["judgement"] = full_df.apply(judge, axis=1)

    assessment = answer_judgements\
        .groupby(['voter_id', '特征'])['judgement']\
        .agg(lambda x: x.mean() * 100)\
        .unstack(level="特征")
    
    assessment.to_csv("result/assessment.csv")
    return assessment


def cluster_students(df, n_clusters):
    agg_cluster = AgglomerativeClustering(n_clusters=n_clusters)
    return agg_cluster.fit_predict(df)


def plot_clusters(df):
    Z = linkage(df, method='ward')

    # 绘制层次聚类的树状图
    plt.figure(figsize=(10, 5))
    dendrogram(Z)
    plt.title('Hierarchical Clustering Dendrogram')
    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.savefig("result/cluster.png")


def import_table_to_pd(table_name):
    # 连接到 PostgreSQL 数据库
    db_connection_info = DBConnectionInfo()
    conn = psycopg2.connect(host=db_connection_info.host, port=db_connection_info.port, database=db_connection_info.database,
                            user=db_connection_info.user, password=db_connection_info.password)
    cursor = conn.cursor()

    # 查询表的字段信息
    cursor.execute(f"SELECT column_name \
            FROM information_schema.columns \
            WHERE table_name = '{table_name}';")
    columns = [row[0] for row in cursor.fetchall()]

    # 查询表的数据
    cursor.execute(f"SELECT {', '.join(columns)} \
            FROM {table_name};")
    rows = cursor.fetchall()

    # 将数据转换为字典列表
    df = pd.DataFrame(rows, columns=columns)

    # 关闭数据库连接
    cursor.close()
    conn.close()

    return df

def export_table_to_json(host, port, database,
                         user, password,
                         table_name, output_file):
    # 连接到 PostgreSQL 数据库
    conn = psycopg2.connect(host=host, port=port, database=database,
                            user=user, password=password)
    cursor = conn.cursor()

    # 查询表的字段信息
    cursor.execute(f"SELECT column_name \
            FROM information_schema.columns \
            WHERE table_name = '{table_name}';")
    columns = [row[0] for row in cursor.fetchall()]

    # 查询表的数据
    cursor.execute(f"SELECT {', '.join(columns)} \
            FROM {table_name};")
    rows = cursor.fetchall()

    # 将数据转换为字典列表
    data = []
    for row in rows:
        data.append(dict(zip(columns, row)))

    def default_serializer(obj):
        if isinstance(obj, datetime):
            return obj.isoformat()
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")


    # 将数据导出为 JSON 格式
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=4, default=default_serializer)

    # 关闭数据库连接
    cursor.close()
    conn.close()


# 示例用法
if __name__ == "__main__":
    student_answers = import_student_answer()
    question_answers = import_question_answers("QnA/")
    assessment = assess_students(student_answers, question_answers)
    plot_clusters(assessment)