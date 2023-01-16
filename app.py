import streamlit as st
import matplotlib.pyplot as plt
from PIL import Image
from model import predict

st.set_option("deprecation.showfileUploaderEncoding", False)

st.title("画像認識Webアプリケーション")
st.write("ディープラーニングで画像の中に何が写っているか認識することができます．")

st.write("")

img_source = st.radio("画像のソースを選択",
                              ("画像アップロード", "カメラ撮影"))
if img_source == "画像アップロード":
    img_file = st.file_uploader("画像を選択してください．", type=["png", "jpg"])
elif img_source == "カメラ撮影":
    img_file = st.camera_input("カメラ撮影")

if img_file is not None:
    with st.spinner("AIで推定中..."):
        img = Image.open(img_file)
        st.image(img, caption="対象の画像", width=480)
        st.write("")

        results = predict(img)

        st.subheader("判定結果")
        n_top = 5  
        for result in results[:n_top]:
            st.write(str(round(result[1]*100, 2)) + "%の確率で" + result[0] + "です．")

        pie_labels = [result[0] for result in results[:n_top]]
        pie_labels.append("others")
        pie_probs = [result[1] for result in results[:n_top]]
        pie_probs.append(sum([result[1] for result in results[n_top:]]))
        fig, ax = plt.subplots()
        wedgeprops={"width":0.3, "edgecolor":"white"}
        textprops = {"fontsize":6}
        ax.pie(pie_probs, labels=pie_labels, counterclock=False, startangle=90,
               textprops=textprops, autopct="%.2f", wedgeprops=wedgeprops)  
        st.pyplot(fig)
