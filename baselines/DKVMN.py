import torch
import torch.nn as nn
import torch.nn.functional as F


class Cell(nn.Module):
    def __init__(self, memory_size, memory_state_dim):
        super(Cell, self).__init__()
        self.memory_size = memory_size
        self.memory_state_dim = memory_state_dim

    def addressing(self, control_input, memory):
        """
        Parameters
        ----------
        control_input: tensor
            embedding vector of input exercise, shape = (batch_size, control_state_dim)
        memory: tensor
            key memory, shape = (memory_size, memory_state_dim)
        Returns
        -------
        correlation_weight: tensor
            correlation weight, shape = (batch_size, memory_size)
        """
        similarity_score = torch.matmul(control_input, torch.t(memory))
        correlation_weight = F.softmax(
            similarity_score, dim=1
        )  # Shape: (batch_size, memory_size)
        return correlation_weight

    def read(self, memory, read_weight):
        """
        Parameters
        ----------
        memory: tensor
            value memory, shape = (batch_size, memory_size, memory_state_dim)
        read_weight: tensor
            correlation weight, shape = (batch_size, memory_size)
        Returns
        -------
        read_content: tensor
            read content, shape = (batch_size, memory_size)
        """
        read_weight = read_weight.view(-1, 1)
        memory = memory.view(-1, self.memory_state_dim)
        rc = torch.mul(read_weight, memory)
        read_content = rc.view(-1, self.memory_size, self.memory_state_dim)
        read_content = torch.sum(read_content, dim=1)
        return read_content


class WriteCell(Cell):
    def __init__(self, memory_size, memory_state_dim):
        super(WriteCell, self).__init__(memory_size, memory_state_dim)
        self.erase = torch.nn.Linear(memory_state_dim, memory_state_dim, bias=True)
        self.add = torch.nn.Linear(memory_state_dim, memory_state_dim, bias=True)
        nn.init.kaiming_normal_(self.erase.weight)
        nn.init.kaiming_normal_(self.add.weight)
        nn.init.constant_(self.erase.bias, 0)
        nn.init.constant_(self.add.bias, 0)

    def write(self, control_input, memory, write_weight):
        """
        Parameters
        ----------
        control_input: tensor
            embedding vector of input exercise and students' answer, shape = (batch_size, control_state_dim)
        memory: tensor
            value memory, shape = (batch_size, memory_size, memory_state_dim)
        read_weight: tensor
            correlation weight, shape = (batch_size, memory_size)
        Returns
        -------
        new_memory: tensor
            updated value memory, shape = (batch_size, memory_size, memory_state_dim)
        """
        erase_signal = torch.sigmoid(self.erase(control_input))
        add_signal = torch.tanh(self.add(control_input))
        erase_reshape = erase_signal.view(-1, 1, self.memory_state_dim)
        add_reshape = add_signal.view(-1, 1, self.memory_state_dim)
        write_weight_reshape = write_weight.view(-1, self.memory_size, 1)
        erase_mult = torch.mul(erase_reshape, write_weight_reshape)
        add_mul = torch.mul(add_reshape, write_weight_reshape)
        new_memory = memory * (1 - erase_mult) + add_mul
        return new_memory


class DKVMNCell(nn.Module):
    def __init__(
        self, memory_size, key_memory_state_dim, value_memory_state_dim, init_key_memory
    ):
        super(DKVMNCell, self).__init__()
        """
        Parameters
        ----------
        memory_size: int
            size of memory
        key_memory_state_dim: int
            dimension of key memory
        value_memory_state_dim:  int
            dimension of value memory
        init_key_memory: tensor
            intial key memory
        """
        self.memory_size = memory_size
        self.key_memory_state_dim = key_memory_state_dim
        self.value_memory_state_dim = value_memory_state_dim

        self.key_head = Cell(
            memory_size=self.memory_size, memory_state_dim=self.key_memory_state_dim
        )
        self.value_head = WriteCell(
            memory_size=self.memory_size, memory_state_dim=self.value_memory_state_dim
        )

        self.key_memory = init_key_memory
        self.value_memory = None

    def init_value_memory(self, value_memory):
        self.value_memory = value_memory

    def attention(self, control_input):
        correlation_weight = self.key_head.addressing(
            control_input=control_input, memory=self.key_memory
        )
        return correlation_weight

    def read(self, read_weight):
        read_content = self.value_head.read(
            memory=self.value_memory, read_weight=read_weight
        )
        return read_content

    def write(self, write_weight, control_input):
        value_memory = self.value_head.write(
            control_input=control_input,
            memory=self.value_memory,
            write_weight=write_weight,
        )
        self.value_memory = nn.Parameter(value_memory.data)

        return self.value_memory


class DKVMN(nn.Module):
    def __init__(
        self,
        n_question,
        batch_size,
        key_embedding_dim=50,
        value_embedding_dim=200,
        memory_size=20,
        key_memory_state_dim=50,
        value_memory_state_dim=200,
        final_fc_dim=50,
        student_num=None,
    ):
        super().__init__()
        self.n_question = n_question
        self.batch_size = batch_size
        self.key_embedding_dim = key_embedding_dim
        self.value_embedding_dim = value_embedding_dim
        self.memory_size = memory_size
        self.key_memory_state_dim = key_memory_state_dim
        self.value_memory_state_dim = value_memory_state_dim
        self.final_fc_dim = final_fc_dim
        self.student_num = student_num

        self.input_embed_linear = nn.Linear(
            self.key_embedding_dim, self.final_fc_dim, bias=True
        )
        self.read_embed_linear = nn.Linear(
            self.value_memory_state_dim + self.final_fc_dim,
            self.final_fc_dim,
            bias=True,
        )
        self.predict_linear = nn.Linear(self.final_fc_dim, 1, bias=True)
        self.init_key_memory = nn.Parameter(
            torch.randn(self.memory_size, self.key_memory_state_dim)
        )
        nn.init.kaiming_normal_(self.init_key_memory)
        self.init_value_memory = nn.Parameter(
            torch.randn(self.memory_size, self.value_memory_state_dim)
        )
        nn.init.kaiming_normal_(self.init_value_memory)

        self.mem = DKVMNCell(
            memory_size=self.memory_size,
            key_memory_state_dim=self.key_memory_state_dim,
            value_memory_state_dim=self.value_memory_state_dim,
            init_key_memory=self.init_key_memory,
        )

        value_memory = nn.Parameter(
            torch.cat(
                [self.init_value_memory.unsqueeze(0) for _ in range(batch_size)], 0
            ).data
        )
        self.mem.init_value_memory(value_memory)

        self.q_embed = nn.Embedding(
            self.n_question + 1, self.key_embedding_dim, padding_idx=0
        )
        self.qa_embed = nn.Embedding(
            2 * self.n_question + 1, self.value_embedding_dim, padding_idx=0
        )

    def init_params(self):
        nn.init.kaiming_normal_(self.predict_linear.weight)
        nn.init.kaiming_normal_(self.read_embed_linear.weight)
        nn.init.constant_(self.read_embed_linear.bias, 0)
        nn.init.constant_(self.predict_linear.bias, 0)

    def init_embeddings(self):
        nn.init.kaiming_normal_(self.q_embed.weight)
        nn.init.kaiming_normal_(self.qa_embed.weight)

    def forward(self, q_embed_data, qa_embed_data):
        batch_size = q_embed_data.shape[0]
        seqlen = q_embed_data.shape[1]

        value_memory = nn.Parameter(
            torch.cat(
                [self.init_value_memory.unsqueeze(0) for _ in range(batch_size)], 0
            ).data
        )
        self.mem.init_value_memory(value_memory)

        slice_q_embed_data = torch.chunk(q_embed_data, seqlen, 1)
        slice_qa_embed_data = torch.chunk(qa_embed_data, seqlen, 1)

        value_read_content_l = []
        input_embed_l = []
        for i in range(seqlen):
            # Attention
            q = slice_q_embed_data[i].squeeze(1)
            correlation_weight = self.mem.attention(q)

            # Read Process
            read_content = self.mem.read(correlation_weight)
            value_read_content_l.append(read_content)
            input_embed_l.append(q)
            # Write Process
            qa = slice_qa_embed_data[i].squeeze(1)
            self.mem.write(correlation_weight, qa)

        all_read_value_content = torch.cat(
            [value_read_content_l[i].unsqueeze(1) for i in range(seqlen)], 1
        )
        input_embed_content = torch.cat(
            [input_embed_l[i].unsqueeze(1) for i in range(seqlen)], 1
        )

        predict_input = torch.cat([all_read_value_content, input_embed_content], 2)
        read_content_embed = torch.tanh(
            self.read_embed_linear(predict_input.view(batch_size * seqlen, -1))
        ).view(batch_size, seqlen, -1)

        return read_content_embed

    def predict(self, q, s, pid=None):
        assert pid is None, "DKVMN does not support pid input"
        q = q.masked_fill(q < 0, 0)
        s = s.masked_fill(s < 0, 0)
        qa = q + s * self.n_question
        q_embed_data = self.q_embed(q)
        qa_embed_data = self.qa_embed(qa)
        h = self(q_embed_data, qa_embed_data)
        return self.predict_linear(h).squeeze(-1), h

    def get_loss(self, q, s, pid=None):
        assert pid is None, "DKVMN does not support pid input"
        logits, _ = self.predict(q, s)
        masked_labels = s[s >= 0].float()
        masked_logits = logits[s >= 0]
        return F.binary_cross_entropy_with_logits(masked_logits, masked_labels)
