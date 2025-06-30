# loocv_resnet50
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, roc_auc_score, matthews_corrcoef, \
    f1_score, precision_score, recall_score
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dense, BatchNormalization, Activation, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime
import gc
import pandas as pd

# ===============================
# CONFIGURAÇÃO CENTRALIZADA
# ===============================
CONFIG = {
    "DATASET_PATH": "data_set_cancer_pancreas",
    "RESULTS_DIR": "resultados",
    "IMG_SIZE": (256, 256),
    "NUM_CLASSES": 2,
    "BATCH_SIZE": 8,
    "EPOCHS": 15,
    "USE_IMAGENET": True,
    "FREEZE_LAYERS": 140,
    "USE_DATA_AUG": False,
    "USE_EARLY_STOPPING": True,
    "START_FOLD": 0,
    "END_FOLD": None,
    "SEED": 42,
    "USE_CLASS_WEIGHTS": False,
    "CLASS_LABELS": ["Negativo para Câncer", "Positivo para Câncer"]
}

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # 0 = all logs, 1 = filter INFO, 2 = filter WARNING, 3 = filter ERROR
tf.keras.utils.set_random_seed(CONFIG["SEED"])
tf.config.experimental.enable_op_determinism()

# Cria diretório com data/hora
timestamp = datetime.now().strftime("%d%m%Y_%H%M")
RESULT_PATH = os.path.join(CONFIG["RESULTS_DIR"], timestamp)
os.makedirs(RESULT_PATH, exist_ok=True)


# ===============================
# FUNÇÕES
# ===============================
def carregar_imagens(data_dir):
    imagens, rotulos, caminhos = [], [], []
    classes = sorted(os.listdir(data_dir))
    for label, classe in enumerate(classes):
        pasta_classe = os.path.join(data_dir, classe)
        for nome_img in os.listdir(pasta_classe):
            caminho_img = os.path.join(pasta_classe, nome_img)
            img = tf.keras.preprocessing.image.load_img(caminho_img, target_size=CONFIG["IMG_SIZE"])
            img_array = tf.keras.preprocessing.image.img_to_array(img) / 255.0
            imagens.append(img_array)
            rotulos.append(label)
            caminhos.append(nome_img)
    return np.array(imagens), np.array(rotulos), CONFIG["CLASS_LABELS"], caminhos


def criar_modelo():
    inputs = Input(shape=(*CONFIG["IMG_SIZE"], 3))
    base_model = tf.keras.applications.ResNet50V2(
        weights='imagenet' if CONFIG["USE_IMAGENET"] else None,
        include_top=False,
        input_tensor=inputs
    )
    print(f"Número total de camadas no backbone: {len(base_model.layers)}")

    if CONFIG["FREEZE_LAYERS"] == -1:
        base_model.trainable = False
    else:
        for layer in base_model.layers[:CONFIG["FREEZE_LAYERS"]]:
            layer.trainable = False

    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(256, use_bias=False, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(128, use_bias=False, kernel_regularizer=regularizers.l2(1e-4))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)
    output = Dense(CONFIG["NUM_CLASSES"], activation='softmax')(x)
    model = Model(inputs=inputs, outputs=output)
    return model


def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.1)
    return img, label


# ===============================
# EXECUÇÃO DO LOOCV
# ===============================
X, y, classes, caminhos = carregar_imagens(CONFIG["DATASET_PATH"])
total_samples = len(X)

start_fold = CONFIG["START_FOLD"]
end_fold = CONFIG["END_FOLD"] or total_samples

model = criar_modelo()
initial_weights = model.get_weights()

results, y_true_all, y_pred_all, y_probs_all = [], [], [], []

if CONFIG["USE_CLASS_WEIGHTS"]:
    class_weights = dict(enumerate((1 / np.bincount(y)) * len(y) / CONFIG["NUM_CLASSES"]))
else:
    class_weights = None

optimizer = optimizers.RMSprop(learning_rate=1e-4)
loss_function = 'categorical_crossentropy'
model.compile(
    loss=loss_function,
    optimizer=optimizer,
    metrics=['accuracy']
)

for i in range(start_fold, end_fold):
    print(f"Fold {i + 1}/{total_samples}")
    model.set_weights(initial_weights)

    train_mask = np.ones(total_samples, dtype=bool)
    train_mask[i] = False

    X_train, y_train = X[train_mask], y[train_mask]
    X_test, y_test = X[i:i + 1], y[i:i + 1]
    y_train_cat = to_categorical(y_train, CONFIG["NUM_CLASSES"])
    y_test_cat = to_categorical(y_test, CONFIG["NUM_CLASSES"])

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train_cat))
    if CONFIG["USE_DATA_AUG"]:
        train_ds = train_ds.map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.shuffle(len(X_train), seed=CONFIG["SEED"]).batch(CONFIG["BATCH_SIZE"]).prefetch(
        tf.data.AUTOTUNE)

    callbacks = []
    if CONFIG["USE_EARLY_STOPPING"]:
        callbacks.append(EarlyStopping(monitor='loss', patience=5, restore_best_weights=True))

    model.fit(train_ds, epochs=CONFIG["EPOCHS"], verbose=0, callbacks=callbacks, class_weight=class_weights)

    pred_probs = model.predict(X_test, verbose=0)
    pred_class = np.argmax(pred_probs)
    true_class = y_test[0]
    accuracy = int(pred_class == true_class)

    fold_result = {
        'Fold': i + 1,
        'Verdadeiro': CONFIG["CLASS_LABELS"][int(true_class)],
        'Predito': CONFIG["CLASS_LABELS"][int(pred_class)],
        'Acurado': accuracy,
        CONFIG["CLASS_LABELS"][0]: pred_probs[0][0],
        CONFIG["CLASS_LABELS"][1]: pred_probs[0][1],
        'Arquivo': caminhos[i]
    }
    results.append(fold_result)
    y_true_all.append(true_class)
    y_pred_all.append(pred_class)
    y_probs_all.append(pred_probs[0][1])

    print(
        f"  Verdadeiro: {CONFIG['CLASS_LABELS'][true_class]}, Predito: {CONFIG['CLASS_LABELS'][pred_class]}, Acurado: {accuracy}")
    print(f"  Acurácia parcial: {np.mean([r['Acurado'] for r in results]) * 100:.2f}%\n")
    gc.collect()

# ===============================
# ANÁLISE FINAL E RELATÓRIOS
# ===============================
# --- PARTE 1: PREPARAÇÃO DOS 3 DATAFRAMES PARA O RELATÓRIO ---

# DataFrame 1: Relatório Detalhado
cm = confusion_matrix(y_true_all, y_pred_all)
report_dict_detailed = classification_report(
    y_true_all, y_pred_all,
    target_names=CONFIG["CLASS_LABELS"],
    output_dict=True,
    zero_division=0
)
report_df_detailed = pd.DataFrame(report_dict_detailed).T
report_df_detailed = report_df_detailed.rename(columns={'f1-score': 'F1-Score'})

# DataFrame 2: Resumo Global
accuracy_global = np.mean([r['Acurado'] for r in results])
auc_global = roc_auc_score(y_true_all, y_probs_all) if len(set(y_true_all)) == 2 else float('nan')
mcc_global = matthews_corrcoef(y_true_all, y_pred_all)
precision_global = precision_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
recall_global = recall_score(y_true_all, y_pred_all, average='weighted', zero_division=0)
f1_global = f1_score(y_true_all, y_pred_all, average='weighted', zero_division=0)

summary_data = {
    'Model': ['ResNet50V2_LOOCV'],
    'AUC': [auc_global], 'CA': [accuracy_global], 'F1': [f1_global],
    'Prec': [precision_global], 'Recall': [recall_global], 'MCC': [mcc_global]
}
summary_df = pd.DataFrame(summary_data).set_index('Model')

# DataFrame 3: Resultados por Fold
df_resultados_por_fold = pd.DataFrame(results)

# --- PARTE 2: EXIBIÇÃO NO CONSOLE E SALVAMENTO EM ARQUIVO ÚNICO COM 3 ABAS ---

# Exibe os relatórios no console
print("\n=============================================")
print("RESUMO GLOBAL DE PERFORMANCE DO MODELO")
print("=============================================")
print(summary_df.to_string(float_format="%.5f"))
print("\n\n=============================================")
print("RELATÓRIO DETALHADO POR CLASSE")
print("=============================================")
print(report_df_detailed.to_string(float_format="%.5f"))
print("=============================================\n")

# Salva os 3 DataFrames em um único arquivo .xlsx com três abas
output_filename = os.path.join(RESULT_PATH, "relatorio_consolidado.xlsx")
with pd.ExcelWriter(output_filename) as writer:
    summary_df.to_excel(writer, sheet_name='Resumo_Global', index=True)
    report_df_detailed.to_excel(writer, sheet_name='Relatorio_Detalhado', index=True)
    df_resultados_por_fold.to_excel(writer, sheet_name='Resultados_por_Fold', index=False)

print(f"Todos os 3 relatórios foram salvos em um único arquivo: {output_filename}")

# ===============================
# SALVAR CONFIGURAÇÕES E GRÁFICOS
# ===============================
# Salva o arquivo de configurações detalhadas
with open(os.path.join(RESULT_PATH, 'configuracoes.txt'), 'w', encoding="utf-8") as f:
    f.write("# ==========================================\n")
    f.write("# CONFIGURAÇÕES DA EXECUÇÃO\n")
    f.write("# ==========================================\n")
    for key, value in CONFIG.items():
        f.write(f"{key}: {value}\n")

    f.write("\n# ==========================================\n")
    f.write("# ARQUITETURA DO MODELO\n")
    f.write("# ==========================================\n")
    f.write("Backbone: ResNet50V2\n\n")
    f.write("# Configuração da Cabeça (Head) do Modelo\n")
    f.write("GlobalAveragePooling2D()\n")
    f.write("Dense(256, kernel_regularizer=l2(1e-4)) -> BatchNormalization -> ReLU -> Dropout(0.3)\n")
    f.write("Dense(128, kernel_regularizer=l2(1e-4)) -> BatchNormalization -> ReLU -> Dropout(0.2)\n")
    f.write(f"Dense({CONFIG['NUM_CLASSES']}) -> Softmax\n")

    f.write("\n# ==========================================\n")
    f.write("# CONFIGURAÇÕES DE COMPILAÇÃO E TREINAMENTO\n")
    f.write("# ==========================================\n")
    f.write(f"model.compile(\n")
    f.write(f"    loss='{loss_function}',\n")
    f.write(f"    optimizer=RMSprop(learning_rate={optimizer.learning_rate.numpy()}),\n")
    f.write(f"    metrics=['accuracy']\n")
    f.write(f")\n\n")

    f.write(f"model.fit(\n")
    f.write(f"    epochs={CONFIG['EPOCHS']},\n")
    if callbacks:
        f.write(f"    callbacks=[EarlyStopping(monitor='loss', patience=5, restore_best_weights=True)],\n")
    else:
        f.write("    callbacks=[],\n")
    f.write(f"    class_weight={'Calculado dinamicamente' if CONFIG['USE_CLASS_WEIGHTS'] else 'None'}\n")
    f.write(f")\n")

# Salva a Matriz de Confusão
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CONFIG["CLASS_LABELS"],
            yticklabels=CONFIG["CLASS_LABELS"])
plt.xlabel("Predito")
plt.ylabel("Verdadeiro")
plt.title("Matriz de Confusão")
plt.tight_layout()
plt.savefig(os.path.join(RESULT_PATH, "matriz_confusao.png"))
plt.close()

# Salva a Curva ROC
if len(set(y_true_all)) == 2:
    fpr, tpr, _ = roc_curve(y_true_all, y_probs_all)
    plt.figure()
    plt.plot(fpr, tpr, label=f'Curva ROC (AUC = {auc_global:.2f})')
    plt.plot([0, 1], [0, 1], linestyle='--')
    plt.xlabel('Taxa de Falsos Positivos (FPR)')
    plt.ylabel('Taxa de Verdadeiros Positivos (TPR)')
    plt.title('Curva ROC')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULT_PATH, "curva_roc.png"))
    plt.close()

print(f"\nExecução finalizada com sucesso. Resultados salvos na pasta: {RESULT_PATH}")