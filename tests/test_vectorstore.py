from fdua_competition.embeddings import create_embeddings
from fdua_competition.enums import VectorStoreOpt
from fdua_competition.vectorstore import FduaVectorStore


class TestFduaVectorStore:
    def setup_method(self):
        embeddings = create_embeddings()
        self.vs = FduaVectorStore(embeddings, VectorStoreOpt.CHROMA)

    def test_create_fdua_vector_store(self):
        assert isinstance(self.vs, FduaVectorStore)
