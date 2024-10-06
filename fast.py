from sentence_transformers import SentenceTransformer
import litserve as ls

class EmbeddingAPI(ls.LitAPI):
    def setup(self, device):
        self.instruction = "Represent this sentence for searching relevant passages: "
        self.model = SentenceTransformer('BAAI/bge-large-en-v1.5', device=device)

    def decode_request(self, request):
        return request["input"]
        
    def batch(self, x):
        return list(x)

    def predict(self, queries):
        return self.model.encode([self.instruction + query for query in queries], normalize_embeddings=True)

    def unbatch(self, outputs):
        return outputs.tolist()

    def encode_response(self, output):
        return {"embedding": output}

if __name__ == "__main__":
    api = EmbeddingAPI()
    server = ls.LitServer(api, max_batch_size=8, batch_timeout=0.01)
    server.run(port=8000)
