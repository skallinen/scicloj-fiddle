(ns scicloj-fiddle.core)

(comment

  (require '[clojure.java.shell :refer [sh]])
  (sh "bash" "-c" "kill $(ps aux | grep 'library(Rserve)' | awk '{print $2}')")

  
  ;;(require '[clojisr.v1.session :as session])
  ;;(swap! clojisr.v1.session/defaults assoc :port 1244)


  (require '[clojisr.v1.r :as r :refer
             [r eval-r->java r->java java->r java->clj java->naive-clj clj->java
              r->clj clj->r ->code r+ colon]]
           '[clojisr.v1.require :refer [require-r]]
           '[clojisr.v1.robject :as robject]
           '[clojisr.v1.rserve :as rserve])

  
  (require-r 
   '[ggplot2 :refer :all]
   '[utils :refer [read-csv]]
   '[grDevices :refer [X11 dev.off]]) 
  ;; '[base :refer
  ;;   [round names ! set.seed sum which rnorm lapply sapply %in% table
  ;;    list.files c paste colnames row.names cbind gsub <- $ $<-
  ;;    as.data.frame data.frame nlevels factor expression is.na strsplit
  ;;    as.character summary]]
  ;; '[graphics :refer [par plot hist legend]]

  
  (require
   '[libpython-clj.python :as py :refer [py. py.. py.-]]
   '[libpython-clj.require :refer [require-python]]
   '[tech.v2.tensor :as t]
   '[tech.ml.dataset :as d]
   '[tech.v2.datatype.functional :as ds])
  ;; '[tech.v2.datatype :as dt] 
  
  ;; (d/column-names reviews-summary)
  ;; (d/ds-row-count reviews-summary)

  (require-python '[nltk :as nltk]  
                  '[torch :as torch]
                  '[numpy :as numpy]
                  '[transformers :as transformers]
                  '[umap :as umap])
  

  (def tokenizer (py. transformers/GPT2Tokenizer "from_pretrained" "gpt2"))

  
  (def model-op (py/$a transformers/GPT2Model from_pretrained "gpt2"))

  
  (py. model-op eval)

  
  (defn sentence-embedding [tokenizer model text]
    (->> text
         (py/$a tokenizer encode)
         (take 100)
         (#(torch/tensor [%]))
         model
         first
         (#(py.. % detach numpy))
         t/->tensor
         first
         (#(ds// (reduce ds/+ %)
                 (first (:shape (t/tensor->dimensions %)))))))
  
  (def reviews-summary
    (->>
     (r->clj (read-csv "/home/voyager/scicloj-fiddle/resources/reviews_summary.csv"))
     (#(d/column % :comments))
     (into [])))

  (def gpt2-embeddings
    (->> comments
         (take 750)
         (map clojure.string/lower-case)
         (map (partial sentence-embedding tokenizer model-op))
         (into [])
         (py/->numpy)))
  
  (def dim-reducer (umap/UMAP))

  (def embedding (py. dim-reducer fit_transform (py.- gpt2-embeddings data)))

  ;; (py.- embedding shape)

  ;; start the graphic device
  (X11 :width 7 :height 7.5)

  (require '[clojisr.v1.applications.plotting :refer
             [plot->svg plot->file plot->buffered-image]])

  (plot->svg
   (let [em  (t/->jvm embedding)
         x (map first em)
         y (map last em)]
     (-> {:x x, :y y}
         d/name-values-seq->dataset
         (ggplot (aes :x x :y y))
         (r+ (geom_point :size 0.1) (xlab "x") (ylab "y")))))
  



  
  ;; start the graphic device
  (X11 :width 7 :height 7.5)
  ;; close a graphic device
  (dev-off)
  (+ 1 1)
  (r "R.Version()")

  (plot :x (repeatedly 100 #(rand-int 100)))

  (plot :x (repeatedly 100 #(rand-int 100)) :type "l" )


  


  (clojure.repl/doc dtype-fn/mean)
  (clojure.repl/dir dtype-fn)
  (clojure.repl/dir dtype-fn)
  (clojure.repl/dir dt)
  (clojure.repl/dir py)

  ;; https://github.com/huggingface/transformers/issues/1528
  ;; tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
  ;; model = GPT2LMHeadModel.from_pretrained('gpt2',output_hidden_states=True)
  ;; model.eval())
  ;; input_ids = torch.tensor(tokenizer.encode("Hello, my dog is cute")).unsqueeze(0) # Batch size 1
  ;; outputs = model(input_ids, labels=input_ids)
  ;; hidden_states = outputs[3]

  ;; https://huggingface.co/transformers/_modules/transformers/modeling_gpt2.html#GPT2Model

  )
