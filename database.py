"""
=============================================================================
数据库模块 — SQLite 存储信号记录、识别结果和模型性能
=============================================================================
"""
import sqlite3
import os
import time
import numpy as np
from config import DB_PATH


def get_connection():
    """获取数据库连接"""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def init_database():
    """初始化数据库表结构"""
    conn = get_connection()
    cursor = conn.cursor()

    # 信号记录表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS signal_records (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            modulation_type TEXT    NOT NULL,
            snr_db          REAL    NOT NULL,
            sample_rate     REAL    NOT NULL,
            symbol_rate     REAL    NOT NULL,
            n_symbols       INTEGER NOT NULL,
            created_at      TEXT    NOT NULL
        )
    """)

    # 识别结果表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS recognition_results (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            signal_id       INTEGER,
            actual_type     TEXT    NOT NULL,
            predicted_type  TEXT    NOT NULL,
            confidence      REAL,
            correct         INTEGER NOT NULL,
            model_name      TEXT    NOT NULL,
            recognition_time TEXT   NOT NULL,
            FOREIGN KEY (signal_id) REFERENCES signal_records(id)
        )
    """)

    # 模型性能记录表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS model_performance (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name      TEXT    NOT NULL,
            accuracy        REAL    NOT NULL,
            precision_score REAL    NOT NULL,
            recall_score    REAL    NOT NULL,
            f1_score        REAL    NOT NULL,
            n_classes       INTEGER NOT NULL,
            n_train_samples INTEGER NOT NULL,
            n_test_samples  INTEGER NOT NULL,
            train_time_sec  REAL    NOT NULL,
            test_time_ms    REAL    NOT NULL,
            recorded_at     TEXT    NOT NULL
        )
    """)

    # 特征重要性表 (针对树模型)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feature_importance (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            model_name      TEXT    NOT NULL,
            feature_name    TEXT    NOT NULL,
            importance      REAL    NOT NULL,
            recorded_at     TEXT    NOT NULL
        )
    """)

    conn.commit()
    conn.close()
    print(f'数据库初始化完成: {DB_PATH}')

def insert_signal_record(mod_type, snr_db, sample_rate, symbol_rate, n_symbols):
    """插入信号生成记录"""
    conn = get_connection()
    cursor = conn.cursor()
    now = time.strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute("""
        INSERT INTO signal_records (modulation_type, snr_db, sample_rate,
                                     symbol_rate, n_symbols, created_at)
        VALUES (?, ?, ?, ?, ?, ?)
    """, (mod_type, snr_db, sample_rate, symbol_rate, n_symbols, now))
    conn.commit()
    record_id = cursor.lastrowid
    conn.close()
    return record_id

def insert_recognition_result(signal_id, actual_type, predicted_type,
                               confidence, correct, model_name):
    """插入识别结果"""
    conn = get_connection()
    cursor = conn.cursor()
    now = time.strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute("""
        INSERT INTO recognition_results (signal_id, actual_type, predicted_type,
                                          confidence, correct, model_name,
                                          recognition_time)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (signal_id, actual_type, predicted_type, confidence,
          int(correct), model_name, now))
    conn.commit()
    result_id = cursor.lastrowid
    conn.close()
    return result_id


def get_recognition_history(limit=50):
    """获取最近的识别历史"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, actual_type, predicted_type, confidence, correct,
               model_name, recognition_time
        FROM recognition_results
        ORDER BY id DESC LIMIT ?
    """, (limit,))
    rows = cursor.fetchall()
    conn.close()
    return rows


def get_recognition_statistics():
    """获取识别统计信息"""
    conn = get_connection()
    cursor = conn.cursor()

    # 总体准确率
    cursor.execute("""
        SELECT COUNT(*) as total,
               SUM(correct) as correct_count
        FROM recognition_results
    """)
    total, correct = cursor.fetchone()
    overall_acc = correct / total if total > 0 else 0

    # 各调制类型准确率
    cursor.execute("""
        SELECT actual_type,
               COUNT(*) as total,
               SUM(correct) as correct_count,
               AVG(confidence) as avg_confidence
        FROM recognition_results
        GROUP BY actual_type
    """)
    per_type = cursor.fetchall()

    conn.close()
    return {
        'total': total,
        'correct': correct,
        'overall_accuracy': overall_acc,
        'per_type': per_type,
    }

def insert_model_performance(model_name, accuracy, precision_val, recall_val,
                              f1_val, n_classes, n_train, n_test,
                              train_time_sec, test_time_ms):
    """插入模型性能记录"""
    conn = get_connection()
    cursor = conn.cursor()
    now = time.strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute("""
        INSERT INTO model_performance (model_name, accuracy, precision_score,
                                        recall_score, f1_score, n_classes,
                                        n_train_samples, n_test_samples,
                                        train_time_sec, test_time_ms, recorded_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (model_name, accuracy, precision_val, recall_val, f1_val,
          n_classes, n_train, n_test, train_time_sec, test_time_ms, now))
    conn.commit()
    conn.close()


def insert_feature_importance(model_name, feature_names, importances):
    """插入特征重要性记录"""
    conn = get_connection()
    cursor = conn.cursor()
    now = time.strftime('%Y-%m-%d %H:%M:%S')
    for fname, imp in zip(feature_names, importances):
        cursor.execute("""
            INSERT INTO feature_importance (model_name, feature_name,
                                             importance, recorded_at)
            VALUES (?, ?, ?, ?)
        """, (model_name, fname, float(imp), now))
    conn.commit()
    conn.close()


def get_model_performance_history():
    """获取所有模型性能记录"""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT model_name, accuracy, precision_score, recall_score, f1_score,
               n_train_samples, n_test_samples, train_time_sec, recorded_at
        FROM model_performance
        ORDER BY recorded_at DESC
    """)
    rows = cursor.fetchall()
    conn.close()
    return rows

